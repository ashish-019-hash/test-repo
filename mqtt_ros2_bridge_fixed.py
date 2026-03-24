import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import paho.mqtt.client as mqtt
import cv2
import base64
import numpy as np
import json

# --- Configuration Constants ---
# NOTE: Update these values to match your MQTT broker and topics
MQTT_BROKER = "localhost" # e.g., "test.mosquitto.org" or your local IP
MQTT_PORT = 1883
MQTT_IMAGE_TOPIC = "/spot/camera/image_stream"  # Outgoing: camera images to MQTT

MQTT_TRIGGER_TOPIC = "spot/hello"  # Incoming: trigger commands from n8n
MQTT_FRANKA_CONTROL_TOPIC = "franka/control"  # Incoming: Franka arm trigger from n8n command center
ROS_IMAGE_TOPIC='/h1/camera/image_raw'
#ROS_IMAGE_TOPIC = "/camera/front/image"  # Subscribed: camera images from ROS2
ROS_TRIGGER_TOPIC = "/spot/trigger_phase2"  # Published: trigger for Spot phase 2
ROS_FRANKA_TRIGGER_TOPIC = "/franka/trigger"  # Published: trigger for Franka arm in Isaac Sim

class ImageToMqttPublisher(Node):
    """
    ROS 2 Node that provides bidirectional MQTT-ROS2 bridging:
    1. Subscribes to a sensor_msgs/Image topic, converts the image data 
       (JPEG-encoded and Base64-encoded), and publishes it to an MQTT topic.
    2. Subscribes to MQTT topic 'spot/hello' (from n8n) and publishes trigger
       messages to ROS2 topic '/spot/trigger_phase2' for Spot's phase 2 movement.
    3. Subscribes to MQTT topic 'franka/control' (from n8n command center) and
       publishes trigger to ROS2 topic '/franka/trigger' for the Franka arm in Isaac Sim.
    """

    def __init__(self):
        super().__init__('image_to_mqtt_publisher')
        self.get_logger().info('Starting ImageToMqttPublisher node (bidirectional bridge)...')

        # 1. Initialize CvBridge for image conversion
        self.bridge = CvBridge()

        # 2. Setup ROS 2 Subscriber for camera images
        self.subscription = self.create_subscription(
            Image,
            ROS_IMAGE_TOPIC,
            self.image_callback,
            10 # QoS history depth
        )
        self.get_logger().info(f'Subscribed to ROS topic: {ROS_IMAGE_TOPIC}')

        # 3. Setup ROS 2 Publisher for trigger messages (to Spot robot)
        self.trigger_publisher = self.create_publisher(
            String,
            ROS_TRIGGER_TOPIC,
            10
        )
        self.get_logger().info(f'Publishing triggers to ROS topic: {ROS_TRIGGER_TOPIC}')

        # 4. Setup ROS 2 Publisher for Franka arm trigger (from n8n command center)
        self.franka_trigger_publisher = self.create_publisher(
            String,
            ROS_FRANKA_TRIGGER_TOPIC,
            10
        )
        self.get_logger().info(f'Publishing Franka triggers to ROS topic: {ROS_FRANKA_TRIGGER_TOPIC}')

        # 5. Setup MQTT Client
        self.mqtt_client = mqtt.Client(client_id="ros2_image_publisher")
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_disconnect = self.on_disconnect
        self.mqtt_client.on_message = self.on_message  # Add message handler
        
        try:
            self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
            # Start a non-blocking background thread to handle MQTT network operations
            self.mqtt_client.loop_start() 
            self.get_logger().info(f'Attempting to connect to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}')
        except Exception as e:
            self.get_logger().error(f'Failed to connect to MQTT broker: {e}')


    def on_connect(self, client, userdata, flags, rc):
        """Callback for when the MQTT client successfully connects."""
        if rc == 0:
            self.get_logger().info("MQTT connected successfully.")
            # Subscribe to the trigger topics from n8n
            client.subscribe(MQTT_TRIGGER_TOPIC, qos=1)
            self.get_logger().info(f"Subscribed to MQTT topic: {MQTT_TRIGGER_TOPIC}")
            client.subscribe(MQTT_FRANKA_CONTROL_TOPIC, qos=1)
            self.get_logger().info(f"Subscribed to MQTT topic: {MQTT_FRANKA_CONTROL_TOPIC}")
        else:
            self.get_logger().error(f"MQTT connection failed with code {rc}")

    def on_disconnect(self, client, userdata, rc):
        """Callback for when the MQTT client disconnects."""
        self.get_logger().warn(f"MQTT disconnected. Trying to reconnect...")
        # Note: paho-mqtt usually handles auto-reconnect if loop_start is running, 
        # but it's good practice to log this.

    def on_message(self, client, userdata, msg):
        """
        Callback for when an MQTT message is received.
        
        Handles messages from n8n:
        - 'spot/hello' -> forwards to ROS2 '/spot/trigger_phase2'
        - 'franka/control' -> forwards to ROS2 '/franka/trigger'
        """
        topic = msg.topic
        try:
            payload = msg.payload.decode('utf-8')
        except UnicodeDecodeError:
            payload = str(msg.payload)

        self.get_logger().info(f"Received MQTT message on '{topic}': {payload}")

        if topic == MQTT_TRIGGER_TOPIC:
            # Forward the trigger to ROS2 for Spot
            ros_msg = String()
            
            # Try to parse as JSON, otherwise use raw payload
            try:
                data = json.loads(payload)
                if isinstance(data, dict):
                    command = data.get('command', data.get('action', 'trigger'))
                    ros_msg.data = str(command)
                else:
                    ros_msg.data = str(data)
            except json.JSONDecodeError:
                ros_msg.data = payload if payload else "trigger"

            self.trigger_publisher.publish(ros_msg)
            self.get_logger().info(f"Published trigger to ROS2 topic '{ROS_TRIGGER_TOPIC}': {ros_msg.data}")

        elif topic == MQTT_FRANKA_CONTROL_TOPIC:
            # Forward the Franka arm trigger to ROS2 for Isaac Sim
            ros_msg = String()
            ros_msg.data = payload if payload else '{"command":"pick_and_place"}'
            self.franka_trigger_publisher.publish(ros_msg)
            self.get_logger().info(f"Published Franka trigger to ROS2 topic '{ROS_FRANKA_TRIGGER_TOPIC}': {ros_msg.data}")

    def image_callback(self, msg):
        """Callback function executed upon receiving a new Image message."""
        try:
            # Convert ROS Image message to OpenCV image format (BGR8)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Encode the image to JPEG format
            # Quality can be adjusted via 'cv2.IMWRITE_JPEG_QUALITY'
            # Note: 95 is a good balance of size and quality
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95] 
            result, encoded_image = cv2.imencode('.jpg', cv_image, encode_param)

            if not result:
                self.get_logger().error("Could not encode image to JPEG.")
                return

            # Convert the numpy array of bytes to a Base64 string for easy MQTT transmission
            base64_string = base64.b64encode(encoded_image.tobytes()).decode('utf-8')

            # Publish the Base64 string to the MQTT topic
            if self.mqtt_client.is_connected():
                # Publish with QoS 1 (at least once)
                self.mqtt_client.publish(MQTT_IMAGE_TOPIC, base64_string, qos=1)
                self.get_logger().debug(f'Published image (Base64 length: {len(base64_string)}) to MQTT topic: {MQTT_IMAGE_TOPIC}')
            else:
                self.get_logger().warn("MQTT client is disconnected, skipping image publish.")

        except Exception as e:
            self.get_logger().error(f'Error processing image and publishing to MQTT: {e}')

    def destroy_node(self):
        """Clean up resources before the node is destroyed."""
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    image_publisher = ImageToMqttPublisher()

    try:
        # Keep the node alive
        rclpy.spin(image_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up and shutdown
        image_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

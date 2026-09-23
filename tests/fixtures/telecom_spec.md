# Carrier Ethernet Service Specification v2.1

## Revision History

| Version | Date | Author | Change |
| --- | --- | --- | --- |
| 1.0 | 2023-01-15 | J. Smith | Initial release |
| 2.0 | 2024-03-02 | A. Lee | Added UNI section |
| 2.1 | 2025-06-30 | A. Lee | Editorial fixes |

## 1 Introduction

This document specifies the service attributes of an Ethernet Virtual Connection (EVC) delivered by a Service Provider to a Subscriber. The EVC is an association of two or more UNIs that limits the exchange of Service Frames.

Figure 2 – Reference model

## 4 Service Attributes

### 4.3 EVC Service Attributes

| Attribute | Type | Units | M/O/C | Range | Default | Description |
| --- | --- | --- | --- | --- | --- | --- |
| CIR | Integer | Mbps | M | 10–1000 | 100 | Committed Information Rate for the EVC |
| EIR | Integer | Mbps | O | 0–1000 | 0 | Excess Information Rate for the EVC |
| EVC ID | String | | M | | | Unique identifier for the EVC assigned by the Service Provider |
| Service Type | Enum | | M | Point-to-Point, Multipoint-to-Multipoint | Point-to-Point | Connectivity type of the EVC |
| CoS Name | String | | O | | Standard | Class of Service name applied to the EVC |

Table 3 – EVC Service Attributes

## 5 UNI

The UNI has a port speed of 1 Gbps. Each UNI SHALL have a physical medium and a MAC address. There are 12 such UNIs in the reference deployment.

### 5.1 UNI Attributes

| Attribute | Type | Units | M/O/C | Range | Default | Description |
| --- | --- | --- | --- | --- | --- | --- |
| Port Speed | Integer | Mbps | M | 10, 100, 1000, 10000 | 1000 | Physical port speed of the UNI |
| Physical Medium | Enum | | M | Copper, Fibre | Fibre | Physical medium of the UNI |
| MAC Address | String | | M | | | MAC address of the UNI port |

## 6 Subscriber

Each Subscriber SHALL have an IMSI and an MSISDN. The Subscriber's MSISDN is an E.164 number such as +447700900123. The Order is created, submitted and terminated by the operator. Provisioning of the service takes up to 5 working days.

## 7 Service Provider

The Service Provider allocates the EVC ID and maintains the Ethernet Virtual Connection for the duration of the contract.

Page 3 of 3

# Networking Guidelines

This guide covers networking best practices and common patterns for Cool Demo PaaS infrastructure.

## VPC Design

### Basic VPC Structure

```yaml
networking:
  vpc:
    cidr: 10.0.0.0/16
    enable_dns_hostnames: true
    enable_dns_support: true
    tags:
      Name: my-web-app-vpc
      Environment: production
```

### Multi-AZ Subnet Design

```yaml
networking:
  vpc:
    cidr: 10.0.0.0/16
  subnets:
    # Public subnets for load balancers
    - name: public-1
      cidr: 10.0.1.0/24
      availability_zone: us-east-1a
      public: true
    - name: public-2
      cidr: 10.0.2.0/24
      availability_zone: us-east-1b
      public: true
    
    # Private subnets for application services
    - name: private-1
      cidr: 10.0.3.0/24
      availability_zone: us-east-1a
      public: false
    - name: private-2
      cidr: 10.0.4.0/24
      availability_zone: us-east-1b
      public: false
    
    # Database subnets for RDS
    - name: database-1
      cidr: 10.0.5.0/24
      availability_zone: us-east-1a
      public: false
    - name: database-2
      cidr: 10.0.6.0/24
      availability_zone: us-east-1b
      public: false
```

## Security Groups

### Web Application Security Groups

```yaml
security_groups:
  # Load balancer security group
  alb-sg:
    rules:
      - type: ingress
        protocol: tcp
        port: 80
        source: 0.0.0.0/0
        description: "HTTP from anywhere"
      - type: ingress
        protocol: tcp
        port: 443
        source: 0.0.0.0/0
        description: "HTTPS from anywhere"
      - type: egress
        protocol: all
        port: all
        destination: 0.0.0.0/0
        description: "All outbound traffic"

  # Web application security group
  web-sg:
    rules:
      - type: ingress
        protocol: tcp
        port: 3000
        source: alb-sg
        description: "HTTP from load balancer"
      - type: egress
        protocol: all
        port: all
        destination: 0.0.0.0/0
        description: "All outbound traffic"

  # Database security group
  database-sg:
    rules:
      - type: ingress
        protocol: tcp
        port: 5432
        source: web-sg
        description: "PostgreSQL from web servers"
      - type: egress
        protocol: all
        port: all
        destination: 0.0.0.0/0
        description: "All outbound traffic"
```

### Microservices Security Groups

```yaml
security_groups:
  # Load balancer security group
  services-alb-sg:
    rules:
      - type: ingress
        protocol: tcp
        port: 443
        source: 0.0.0.0/0
        description: "HTTPS from anywhere"
      - type: egress
        protocol: all
        port: all
        destination: 0.0.0.0/0
        description: "All outbound traffic"

  # User service security group
  user-service-sg:
    rules:
      - type: ingress
        protocol: tcp
        port: 3000
        source: services-alb-sg
        description: "HTTP from load balancer"
      - type: egress
        protocol: all
        port: all
        destination: 0.0.0.0/0
        description: "All outbound traffic"

  # Order service security group
  order-service-sg:
    rules:
      - type: ingress
        protocol: tcp
        port: 3000
        source: services-alb-sg
        description: "HTTP from load balancer"
      - type: egress
        protocol: all
        port: all
        destination: 0.0.0.0/0
        description: "All outbound traffic"
```

## Load Balancer Configuration

### Application Load Balancer

```yaml
load_balancers:
  web-alb:
    type: alb
    scheme: internet-facing
    subnets:
      - public-1
      - public-2
    security_groups:
      - alb-sg
    listeners:
      - port: 80
        protocol: HTTP
        default_action:
          type: redirect
          redirect_config:
            protocol: HTTPS
            port: 443
            status_code: HTTP_301
      - port: 443
        protocol: HTTPS
        certificate: web-cert
        default_action:
          type: forward
          target_group: web-tg
```

### Internal Load Balancer

```yaml
load_balancers:
  internal-alb:
    type: alb
    scheme: internal
    subnets:
      - private-1
      - private-2
    security_groups:
      - internal-alb-sg
    listeners:
      - port: 80
        protocol: HTTP
        default_action:
          type: forward
          target_group: api-tg
```

## DNS Configuration

### Public DNS

```yaml
dns:
  zones:
    - domain: example.com
      records:
        - name: www
          type: A
          alias: true
          target: web-alb
          evaluate_target_health: true
        - name: api
          type: A
          alias: true
          target: api-alb
          evaluate_target_health: true
```

### Private DNS

```yaml
dns:
  zones:
    - domain: internal.example.com
      private: true
      vpc: my-web-app-vpc
      records:
        - name: database
          type: CNAME
          value: database.cluster-xyz.us-east-1.rds.amazonaws.com
        - name: cache
          type: CNAME
          value: cache.xyz.cache.amazonaws.com
```

## Common Networking Patterns

### 1. Three-Tier Architecture

```yaml
# Web tier (public subnets)
services:
  web-servers:
    type: ec2
    instance_type: t3.small
    subnets:
      - public-1
      - public-2
    security_groups:
      - web-sg

# Application tier (private subnets)
services:
  app-servers:
    type: ecs
    cpu: 512
    memory: 1024
    subnets:
      - private-1
      - private-2
    security_groups:
      - app-sg

# Database tier (database subnets)
services:
  database:
    type: rds
    engine: postgres
    instance_class: db.t3.micro
    subnets:
      - database-1
      - database-2
    security_groups:
      - database-sg
```

### 2. Microservices Architecture

```yaml
# Application Load Balancer for routing
load_balancers:
  services-alb:
    type: alb
    scheme: internet-facing
    subnets:
      - public-1
      - public-2
    listeners:
      - port: 443
        protocol: HTTPS
        certificate: services-cert
        rules:
          - condition:
              path_pattern: "/users/*"
            action:
              type: forward
              target_group: user-service-tg
          - condition:
              path_pattern: "/orders/*"
            action:
              type: forward
              target_group: order-service-tg

# User service
services:
  user-service:
    type: ecs
    cpu: 256
    memory: 512
    subnets:
      - private-1
      - private-2
    security_groups:
      - user-service-sg

# Order service
services:
  order-service:
    type: ecs
    cpu: 256
    memory: 512
    subnets:
      - private-1
      - private-2
    security_groups:
      - order-service-sg
```

## Network Security Best Practices

### 1. Principle of Least Privilege

```yaml
security_groups:
  web-sg:
    rules:
      - type: ingress
        protocol: tcp
        port: 3000
        source: alb-sg
        description: "Only allow traffic from load balancer"
      - type: egress
        protocol: tcp
        port: 443
        destination: 0.0.0.0/0
        description: "Only allow HTTPS outbound"
      - type: egress
        protocol: tcp
        port: 5432
        destination: database-sg
        description: "Only allow database access"
```

### 2. Network Segmentation

```yaml
networking:
  vpc:
    cidr: 10.0.0.0/16
  subnets:
    # Web tier
    - name: web-1
      cidr: 10.0.1.0/24
      availability_zone: us-east-1a
      public: true
    - name: web-2
      cidr: 10.0.2.0/24
      availability_zone: us-east-1b
      public: true
    
    # Application tier
    - name: app-1
      cidr: 10.0.10.0/24
      availability_zone: us-east-1a
      public: false
    - name: app-2
      cidr: 10.0.11.0/24
      availability_zone: us-east-1b
      public: false
    
    # Database tier
    - name: db-1
      cidr: 10.0.20.0/24
      availability_zone: us-east-1a
      public: false
    - name: db-2
      cidr: 10.0.21.0/24
      availability_zone: us-east-1b
      public: false
```

### 3. VPC Flow Logs

```yaml
networking:
  vpc:
    cidr: 10.0.0.0/16
    flow_logs:
      enabled: true
      destination: cloudwatch
      log_group: /aws/vpc/flowlogs
      retention_days: 30
```

## Troubleshooting Common Issues

### 1. Connectivity Issues

**Problem**: Services can't communicate with each other

**Solution**: Check security group rules and subnet routing

```yaml
security_groups:
  web-sg:
    rules:
      - type: egress
        protocol: tcp
        port: 3000
        destination: app-sg
        description: "Allow communication with app tier"
```

### 2. Load Balancer Health Check Failures

**Problem**: Load balancer health checks failing

**Solution**: Ensure health check endpoint is accessible and returns 200

```yaml
services:
  web-app:
    type: ecs
    cpu: 512
    memory: 1024
    image: my-app:latest
    port: 3000
    health_check:
      command: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 30
      timeout: 5
      retries: 3
```

### 3. DNS Resolution Issues

**Problem**: Services can't resolve internal DNS names

**Solution**: Enable DNS resolution in VPC and use private hosted zones

```yaml
networking:
  vpc:
    cidr: 10.0.0.0/16
    enable_dns_hostnames: true
    enable_dns_support: true

dns:
  zones:
    - domain: internal.example.com
      private: true
      vpc: my-web-app-vpc
```

## Summary

Following these networking guidelines will help you build secure, scalable, and maintainable infrastructure:

1. **Design** your VPC with proper subnet segmentation
2. **Secure** your network with appropriate security groups
3. **Load balance** your traffic with ALBs and target groups
4. **Configure** DNS for both public and private resources
5. **Monitor** your network with VPC Flow Logs
6. **Troubleshoot** connectivity issues systematically

For more specific guidance, check out our [Best Practices](/guidelines/best-practices) and [How-tos](/guidelines/how-tos) sections.

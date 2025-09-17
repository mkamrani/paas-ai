# How-tos

This guide provides step-by-step instructions for common tasks with Cool Demo PaaS.

## Getting Started

### 1. Create Your First Infrastructure

Create a simple web application with load balancer:

```yaml
# infrastructure.yaml
project:
  name: my-first-app
  environment: development
  region: us-east-1

# networking.yaml
networking:
  vpc:
    cidr: 10.0.0.0/16
  subnets:
    - name: public-1
      cidr: 10.0.1.0/24
      availability_zone: us-east-1a
      public: true
    - name: private-1
      cidr: 10.0.2.0/24
      availability_zone: us-east-1a
      public: false

# services.yaml
services:
  web-app:
    type: ecs
    cpu: 256
    memory: 512
    image: nginx:latest
    port: 80
    desired_count: 2
    subnets:
      - private-1
    security_groups:
      - web-sg

# load-balancer.yaml
load_balancers:
  web-alb:
    type: alb
    scheme: internet-facing
    subnets:
      - public-1
    listeners:
      - port: 80
        protocol: HTTP
        default_action:
          type: forward
          service: web-app

# security-groups.yaml
security_groups:
  web-sg:
    rules:
      - type: ingress
        protocol: tcp
        port: 80
        source: alb-sg
      - type: egress
        protocol: all
        port: all
        destination: 0.0.0.0/0

  alb-sg:
    rules:
      - type: ingress
        protocol: tcp
        port: 80
        source: 0.0.0.0/0
      - type: egress
        protocol: all
        port: all
        destination: 0.0.0.0/0
```

### 2. Add HTTPS Support

Add SSL certificate and HTTPS listener:

```yaml
# certificates.yaml
certificates:
  web-cert:
    domain: example.com
    subject_alternative_names:
      - www.example.com
    validation_method: DNS
    auto_validation: true

# load-balancer.yaml (updated)
load_balancers:
  web-alb:
    type: alb
    scheme: internet-facing
    subnets:
      - public-1
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
          service: web-app

# dns.yaml
dns:
  zones:
    - domain: example.com
      records:
        - name: www
          type: A
          alias: true
          target: web-alb
          evaluate_target_health: true
```

## Common Use Cases

### 1. Deploy a Node.js Application

```yaml
# services.yaml
services:
  node-app:
    type: ecs
    cpu: 512
    memory: 1024
    image: my-node-app:latest
    port: 3000
    desired_count: 3
    environment:
      - name: NODE_ENV
        value: production
      - name: PORT
        value: "3000"
    secrets:
      - name: DATABASE_URL
        value: "database-connection-string"
      - name: JWT_SECRET
        value: "jwt-secret-key"
    health_check:
      command: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 30
      timeout: 5
      retries: 3
    logging:
      driver: awslogs
      options:
        awslogs-group: /ecs/node-app
        awslogs-region: us-east-1
        awslogs-stream-prefix: ecs
```

### 2. Set Up a Database

```yaml
# database.yaml
services:
  postgres-db:
    type: rds
    engine: postgres
    engine_version: "13.7"
    instance_class: db.t3.micro
    allocated_storage: 20
    backup_retention_period: 7
    backup_window: "03:00-04:00"
    maintenance_window: "sun:04:00-sun:05:00"
    subnets:
      - database-1
      - database-2
    security_groups:
      - database-sg
    parameters:
      - name: shared_preload_libraries
        value: "pg_stat_statements"

# security-groups.yaml (updated)
security_groups:
  database-sg:
    rules:
      - type: ingress
        protocol: tcp
        port: 5432
        source: app-sg
      - type: egress
        protocol: all
        port: all
        destination: 0.0.0.0/0
```

### 3. Create a Microservices Architecture

```yaml
# load-balancer.yaml
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
              service: user-service
          - condition:
              path_pattern: "/orders/*"
            action:
              type: forward
              service: order-service
          - condition:
              path_pattern: "/products/*"
            action:
              type: forward
              service: product-service

# services.yaml
services:
  user-service:
    type: ecs
    cpu: 256
    memory: 512
    image: user-service:latest
    port: 3000
    desired_count: 2
    environment:
      - name: NODE_ENV
        value: production
    secrets:
      - name: DATABASE_URL
        value: "user-db-connection"
    health_check:
      command: ["CMD", "curl", "-f", "http://localhost:3000/health"]

  order-service:
    type: ecs
    cpu: 256
    memory: 512
    image: order-service:latest
    port: 3000
    desired_count: 2
    environment:
      - name: NODE_ENV
        value: production
    secrets:
      - name: DATABASE_URL
        value: "order-db-connection"
    health_check:
      command: ["CMD", "curl", "-f", "http://localhost:3000/health"]

  product-service:
    type: ecs
    cpu: 256
    memory: 512
    image: product-service:latest
    port: 3000
    desired_count: 2
    environment:
      - name: NODE_ENV
        value: production
    secrets:
      - name: DATABASE_URL
        value: "product-db-connection"
    health_check:
      command: ["CMD", "curl", "-f", "http://localhost:3000/health"]


```

### 4. Set Up Auto Scaling

```yaml
# services.yaml
services:
  scalable-app:
    type: ecs
    cpu: 512
    memory: 1024
    image: my-app:latest
    port: 3000
    min_capacity: 2
    max_capacity: 10
    desired_count: 3
    scaling_policies:
      - metric: cpu_utilization
        target_value: 70
        scale_out_cooldown: 300
        scale_in_cooldown: 300
      - metric: memory_utilization
        target_value: 80
        scale_out_cooldown: 300
        scale_in_cooldown: 300
    scaling_policies:
      - metric: request_count
        target_value: 1000
        scale_out_cooldown: 300
        scale_in_cooldown: 300
```

### 5. Implement Blue-Green Deployment

```yaml
# services.yaml
services:
  web-app-blue:
    type: ecs
    cpu: 512
    memory: 1024
    image: my-app:v1.0
    port: 3000
    desired_count: 3

  web-app-green:
    type: ecs
    cpu: 512
    memory: 1024
    image: my-app:v2.0
    port: 3000
    desired_count: 0

# load-balancer.yaml
load_balancers:
  web-alb:
    type: alb
    scheme: internet-facing
    subnets:
      - public-1
      - public-2
    listeners:
      - port: 443
        protocol: HTTPS
        certificate: web-cert
        rules:
          - condition:
              path_pattern: "/*"
            action:
              type: forward
              services:
                - service: web-app-blue
                  weight: 100
                - service: web-app-green
                  weight: 0
```

## Environment Management

### 1. Create Environment-Specific Configurations

```yaml
# production.yaml
project:
  name: my-app
  environment: production
  region: us-east-1

services:
  web-app:
    type: ecs
    cpu: 1024
    memory: 2048
    min_capacity: 3
    max_capacity: 10
    desired_count: 5
    image: my-app:latest
    environment:
      - name: NODE_ENV
        value: production
      - name: LOG_LEVEL
        value: info

# staging.yaml
project:
  name: my-app
  environment: staging
  region: us-east-1

services:
  web-app:
    type: ecs
    cpu: 512
    memory: 1024
    min_capacity: 1
    max_capacity: 3
    desired_count: 2
    image: my-app:staging
    environment:
      - name: NODE_ENV
        value: staging
      - name: LOG_LEVEL
        value: debug

# development.yaml
project:
  name: my-app
  environment: development
  region: us-east-1

services:
  web-app:
    type: ecs
    cpu: 256
    memory: 512
    min_capacity: 1
    max_capacity: 2
    desired_count: 1
    image: my-app:dev
    environment:
      - name: NODE_ENV
        value: development
      - name: LOG_LEVEL
        value: debug
```

### 2. Use Secrets Management

```yaml
# services.yaml
services:
  web-app:
    type: ecs
    cpu: 512
    memory: 1024
    image: my-app:latest
    port: 3000
    desired_count: 3
    secrets:
      - name: DATABASE_URL
        value: "database-connection-string"
      - name: API_KEY
        value: "external-api-key"
      - name: JWT_SECRET
        value: "jwt-secret-key"
```

## Monitoring and Logging

### 1. Set Up CloudWatch Logs

```yaml
# services.yaml
services:
  web-app:
    type: ecs
    cpu: 512
    memory: 1024
    image: my-app:latest
    port: 3000
    desired_count: 3
    logging:
      driver: awslogs
      options:
        awslogs-group: /ecs/web-app
        awslogs-region: us-east-1
        awslogs-stream-prefix: ecs
        awslogs-create-group: true
```

### 2. Configure Monitoring and Alarms

```yaml
# services.yaml
services:
  web-app:
    type: ecs
    cpu: 512
    memory: 1024
    image: my-app:latest
    port: 3000
    desired_count: 3
    monitoring:
      metrics:
        - cpu_utilization
        - memory_utilization
        - request_count
        - response_time
      alarms:
        - metric: cpu_utilization
          threshold: 80
          comparison: GreaterThanThreshold
          period: 300
          evaluation_periods: 2
          alarm_actions:
            - "arn:aws:sns:us-east-1:123456789012:alerts"
        - metric: memory_utilization
          threshold: 85
          comparison: GreaterThanThreshold
          period: 300
          evaluation_periods: 2
          alarm_actions:
            - "arn:aws:sns:us-east-1:123456789012:alerts"
```

## Troubleshooting

### 1. Debug Service Issues

**Check service logs:**
```bash
aws logs describe-log-groups --log-group-name-prefix /ecs/web-app
aws logs get-log-events --log-group-name /ecs/web-app --log-stream-name ecs/web-app/1234567890
```

**Check service status:**
```bash
aws ecs describe-services --cluster my-cluster --services web-app
aws ecs describe-tasks --cluster my-cluster --tasks task-arn
```

### 2. Debug Load Balancer Issues

**Check target group health:**
```bash
aws elbv2 describe-target-health --target-group-arn arn:aws:elasticloadbalancing:us-east-1:123456789012:targetgroup/web-tg/1234567890
```

**Check load balancer logs:**
```bash
aws s3 ls s3://my-alb-logs/
aws s3 cp s3://my-alb-logs/AWSLogs/123456789012/elasticloadbalancing/us-east-1/2023/01/01/123456789012_elasticloadbalancing_us-east-1_app.web-alb.1234567890.us-east-1.elb.amazonaws.com_20230101T0000Z_1.2.3.4_1234567890.log .
```

### 3. Debug DNS Issues

**Check DNS resolution:**
```bash
nslookup www.example.com
dig www.example.com
```

**Check Route53 records:**
```bash
aws route53 list-resource-record-sets --hosted-zone-id Z1234567890
```

## Summary

These how-tos cover the most common scenarios for using Cool Demo PaaS:

1. **Getting Started** - Create your first infrastructure
2. **Common Use Cases** - Deploy applications, databases, and microservices
3. **Environment Management** - Manage different environments
4. **Monitoring and Logging** - Set up observability
5. **Troubleshooting** - Debug common issues

For more detailed guidance, check out our [Best Practices](/guidelines/best-practices) and [Networking Guidelines](/guidelines/networking) sections.

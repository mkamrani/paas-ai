# Best Practices

This guide covers best practices for using Cool Demo PaaS to build reliable, secure, and scalable AWS infrastructure.

## Infrastructure Design

### 1. Use Multi-File Organization

Organize your infrastructure across multiple YAML files for better maintainability:

```yaml
# infrastructure.yaml - Main project configuration
project:
  name: my-web-app
  environment: production
  region: us-east-1

# networking.yaml - Network infrastructure
networking:
  vpc:
    cidr: 10.0.0.0/16
  subnets:
    - name: public-1
      cidr: 10.0.1.0/24
      availability_zone: us-east-1a
    - name: private-1
      cidr: 10.0.2.0/24
      availability_zone: us-east-1a

# services.yaml - Application services
services:
  web-app:
    type: ecs
    cpu: 512
    memory: 1024
    image: my-app:latest
    port: 3000
    desired_count: 3

# load-balancer.yaml - Load balancing configuration
load_balancers:
  web-alb:
    type: alb
    scheme: internet-facing
    listeners:
      - port: 443
        protocol: HTTPS
        certificate: web-cert
        default_action:
          type: forward
          target_group: web-tg

# dns.yaml - DNS configuration
dns:
  zones:
    - domain: example.com
      records:
        - name: www
          type: A
          alias: true
          target: web-alb
```

### 2. Environment-Specific Configurations

Use environment-specific configurations for different deployment stages:

```yaml
# production.yaml
project:
  name: my-web-app
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

# staging.yaml
project:
  name: my-web-app
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
```

## Security Best Practices

### 1. Use Security Groups Effectively

```yaml
security_groups:
  web-sg:
    rules:
      - type: ingress
        protocol: tcp
        port: 80
        source: 0.0.0.0/0
      - type: ingress
        protocol: tcp
        port: 443
        source: 0.0.0.0/0
      - type: egress
        protocol: all
        port: all
        destination: 0.0.0.0/0

  database-sg:
    rules:
      - type: ingress
        protocol: tcp
        port: 5432
        source: web-sg
      - type: egress
        protocol: all
        port: all
        destination: 0.0.0.0/0
```

### 2. Use HTTPS Everywhere

```yaml
load_balancers:
  web-alb:
    type: alb
    scheme: internet-facing
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

### 3. Use Simplified Secrets Management

```yaml
services:
  api:
    type: ecs
    cpu: 512
    memory: 1024
    image: my-api:latest
    secrets:
      - name: DATABASE_URL
        value: "database-connection-string"
      - name: API_KEY
        value: "external-api-key"
```

The PaaS automatically handles:
- Creating secrets in AWS Secrets Manager
- Mapping to correct ARNs
- IAM permissions
- Secret rotation

## Performance Best Practices

### 1. Right-Size Your Resources

```yaml
services:
  # Development
  dev-api:
    type: ecs
    cpu: 256
    memory: 512

  # Production
  prod-api:
    type: ecs
    cpu: 1024
    memory: 2048
    min_capacity: 3
    max_capacity: 10
    desired_capacity: 5
```

### 2. Implement Auto Scaling

```yaml
services:
  web-app:
    type: ecs
    cpu: 512
    memory: 1024
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
```

### 3. Use Health Checks

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

## Monitoring and Logging

### 1. Enable CloudWatch Logs

```yaml
services:
  web-app:
    type: ecs
    cpu: 512
    memory: 1024
    image: my-app:latest
    logging:
      driver: awslogs
      options:
        awslogs-group: /ecs/my-web-app
        awslogs-region: us-east-1
        awslogs-stream-prefix: ecs
```

### 2. Set Up Monitoring

```yaml
services:
  web-app:
    type: ecs
    cpu: 512
    memory: 1024
    image: my-app:latest
    monitoring:
      metrics:
        - cpu_utilization
        - memory_utilization
        - request_count
      alarms:
        - metric: cpu_utilization
          threshold: 80
          comparison: GreaterThanThreshold
          period: 300
          evaluation_periods: 2
```

## Cost Optimization

### 1. Use Spot Instances for Non-Critical Workloads

```yaml
services:
  batch-job:
    type: ecs
    cpu: 1024
    memory: 2048
    spot: true
    max_capacity: 5
```

### 2. Implement Resource Scheduling

```yaml
services:
  dev-api:
    type: ecs
    cpu: 256
    memory: 512
    scheduling:
      start_time: "09:00"
      stop_time: "18:00"
      timezone: "America/New_York"
      weekdays_only: true
```

## Disaster Recovery

### 1. Multi-AZ Deployment

```yaml
networking:
  vpc:
    cidr: 10.0.0.0/16
  subnets:
    - name: public-1
      cidr: 10.0.1.0/24
      availability_zone: us-east-1a
    - name: public-2
      cidr: 10.0.2.0/24
      availability_zone: us-east-1b
    - name: private-1
      cidr: 10.0.3.0/24
      availability_zone: us-east-1a
    - name: private-2
      cidr: 10.0.4.0/24
      availability_zone: us-east-1b
```

### 2. Backup and Recovery

```yaml
services:
  database:
    type: rds
    engine: postgres
    instance_class: db.t3.micro
    backup:
      retention_period: 7
      backup_window: "03:00-04:00"
      maintenance_window: "sun:04:00-sun:05:00"
```

## Common Anti-Patterns to Avoid

### 1. Don't Use Hardcoded Values

❌ **Bad:**
```yaml
services:
  web-app:
    type: ecs
    cpu: 512
    memory: 1024
    image: my-app:latest
    environment:
      - name: DATABASE_URL
        value: "postgresql://user:password@db:5432/mydb"
```

✅ **Good:**
```yaml
services:
  web-app:
    type: ecs
    cpu: 512
    memory: 1024
    image: my-app:latest
    secrets:
      - name: DATABASE_URL
        value: "database-connection-string"
```

### 2. Don't Skip Health Checks

❌ **Bad:**
```yaml
services:
  web-app:
    type: ecs
    cpu: 512
    memory: 1024
    image: my-app:latest
    # No health check
```

✅ **Good:**
```yaml
services:
  web-app:
    type: ecs
    cpu: 512
    memory: 1024
    image: my-app:latest
    health_check:
      command: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 30
      timeout: 5
      retries: 3
```

### 3. Don't Use Single Points of Failure

❌ **Bad:**
```yaml
networking:
  vpc:
    cidr: 10.0.0.0/16
  subnets:
    - name: public-1
      cidr: 10.0.1.0/24
      availability_zone: us-east-1a
    # Only one subnet
```

✅ **Good:**
```yaml
networking:
  vpc:
    cidr: 10.0.0.0/16
  subnets:
    - name: public-1
      cidr: 10.0.1.0/24
      availability_zone: us-east-1a
    - name: public-2
      cidr: 10.0.2.0/24
      availability_zone: us-east-1b
```

## Summary

Following these best practices will help you build reliable, secure, and scalable infrastructure with Cool Demo PaaS. Remember to:

1. **Organize** your infrastructure across multiple files
2. **Secure** your applications with proper security groups and HTTPS
3. **Scale** your applications with auto scaling and health checks
4. **Monitor** your infrastructure with proper logging and metrics
5. **Optimize** costs with appropriate resource sizing and scheduling
6. **Plan** for disaster recovery with multi-AZ deployments

For more specific guidance, check out our [Networking Guidelines](/guidelines/networking) and [How-tos](/guidelines/how-tos) sections.

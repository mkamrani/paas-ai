# EC2 - Elastic Compute Cloud

EC2 provides scalable virtual machines in the cloud. Our PaaS simplifies EC2 configuration by providing sensible defaults and simple options.

## Basic Configuration

```yaml
services:
  web-server:
    type: ec2
    instance_type: t3.micro
    ami: amazon-linux-2
    key_pair: my-key-pair
    security_groups:
      - web-sg
    user_data: |
      #!/bin/bash
      yum update -y
      yum install -y nginx
      systemctl start nginx
      systemctl enable nginx
```

## Configuration Options

### Instance Types

We support the following instance families with common sizes:

| Family | Sizes | Use Case |
|--------|-------|----------|
| t3 | micro, small, medium, large | General purpose, burstable |
| t3a | micro, small, medium, large | General purpose, AMD-based |
| m5 | large, xlarge, 2xlarge | General purpose, balanced |
| c5 | large, xlarge, 2xlarge | Compute optimized |
| r5 | large, xlarge, 2xlarge | Memory optimized |

### AMI Options

- `amazon-linux-2`: Latest Amazon Linux 2
- `ubuntu-20.04`: Ubuntu 20.04 LTS
- `ubuntu-22.04`: Ubuntu 22.04 LTS
- `windows-2019`: Windows Server 2019
- `windows-2022`: Windows Server 2022

### Security Groups

Security groups are automatically created with sensible defaults:

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
```

## Advanced Configuration

### Auto Scaling

```yaml
services:
  web-servers:
    type: ec2
    instance_type: t3.small
    min_capacity: 2
    max_capacity: 10
    desired_capacity: 3
    scaling_policies:
      - metric: cpu_utilization
        target_value: 70
        scale_out_cooldown: 300
        scale_in_cooldown: 300
```

### Load Balancer Integration

```yaml
services:
  web-server:
    type: ec2
    instance_type: t3.medium
    health_check:
      path: /health
      port: 80
      protocol: HTTP
```

## Best Practices

1. **Use appropriate instance types** for your workload
2. **Enable detailed monitoring** for production instances
3. **Use security groups** to restrict access
4. **Implement auto scaling** for variable workloads
5. **Use user data scripts** for consistent configuration

## Example: Complete Web Server Setup

```yaml
# web-servers.yaml
services:
  web-servers:
    type: ec2
    instance_type: t3.small
    ami: amazon-linux-2
    key_pair: production-key
    security_groups:
      - web-sg
    min_capacity: 2
    max_capacity: 8
    desired_count: 3
    user_data: |
      #!/bin/bash
      yum update -y
      yum install -y nginx
      echo "<h1>Hello from $(hostname)</h1>" > /var/www/html/index.html
      systemctl start nginx
      systemctl enable nginx
```

This configuration creates an auto-scaling group of web servers behind a load balancer.

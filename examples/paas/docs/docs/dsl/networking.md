# Networking - VPC and Subnets

Networking in Cool Demo PaaS is designed to be simple yet powerful. The PaaS automatically handles most networking complexity while giving you control over the essential aspects.

## Basic VPC Configuration

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

## Subnet Configuration

### Public Subnets (for Load Balancers)

```yaml
networking:
  vpc:
    cidr: 10.0.0.0/16
  subnets:
    - name: public-1
      cidr: 10.0.1.0/24
      availability_zone: us-east-1a
      public: true
    - name: public-2
      cidr: 10.0.2.0/24
      availability_zone: us-east-1b
      public: true
```

### Private Subnets (for Application Services)

```yaml
networking:
  vpc:
    cidr: 10.0.0.0/16
  subnets:
    - name: private-1
      cidr: 10.0.10.0/24
      availability_zone: us-east-1a
      public: false
    - name: private-2
      cidr: 10.0.11.0/24
      availability_zone: us-east-1b
      public: false
```

### Database Subnets (for RDS)

```yaml
networking:
  vpc:
    cidr: 10.0.0.0/16
  subnets:
    - name: database-1
      cidr: 10.0.20.0/24
      availability_zone: us-east-1a
      public: false
    - name: database-2
      cidr: 10.0.21.0/24
      availability_zone: us-east-1b
      public: false
```

## Complete Multi-AZ Setup

```yaml
networking:
  vpc:
    cidr: 10.0.0.0/16
    enable_dns_hostnames: true
    enable_dns_support: true
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
      cidr: 10.0.10.0/24
      availability_zone: us-east-1a
      public: false
    - name: private-2
      cidr: 10.0.11.0/24
      availability_zone: us-east-1b
      public: false
    
    # Database subnets for RDS
    - name: database-1
      cidr: 10.0.20.0/24
      availability_zone: us-east-1a
      public: false
    - name: database-2
      cidr: 10.0.21.0/24
      availability_zone: us-east-1b
      public: false
```

## Security Groups

### Basic Security Group Configuration

```yaml
security_groups:
  web-sg:
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
```

### Application Security Groups

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

## Default Networking Behavior

If you don't specify networking configuration, the PaaS will automatically create:

### Default VPC Configuration
- CIDR: `10.0.0.0/16`
- DNS hostnames: Enabled
- DNS support: Enabled

### Default Subnets
- 2 public subnets (one per AZ)
- 2 private subnets (one per AZ)
- Automatic CIDR allocation
- Proper routing tables

### Default Security Groups
- Web security group with HTTP/HTTPS ingress
- Database security group with restricted access
- Load balancer security group with public access

## VPC Flow Logs

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

## Common Networking Patterns

### 1. Three-Tier Architecture

```yaml
networking:
  vpc:
    cidr: 10.0.0.0/16
  subnets:
    # Web tier (public)
    - name: web-1
      cidr: 10.0.1.0/24
      availability_zone: us-east-1a
      public: true
    - name: web-2
      cidr: 10.0.2.0/24
      availability_zone: us-east-1b
      public: true
    
    # Application tier (private)
    - name: app-1
      cidr: 10.0.10.0/24
      availability_zone: us-east-1a
      public: false
    - name: app-2
      cidr: 10.0.11.0/24
      availability_zone: us-east-1b
      public: false
    
    # Database tier (private)
    - name: db-1
      cidr: 10.0.20.0/24
      availability_zone: us-east-1a
      public: false
    - name: db-2
      cidr: 10.0.21.0/24
      availability_zone: us-east-1b
      public: false
```

### 2. Microservices Architecture

```yaml
networking:
  vpc:
    cidr: 10.0.0.0/16
  subnets:
    # Public subnets for ALBs
    - name: public-1
      cidr: 10.0.1.0/24
      availability_zone: us-east-1a
      public: true
    - name: public-2
      cidr: 10.0.2.0/24
      availability_zone: us-east-1b
      public: true
    
    # Private subnets for services
    - name: services-1
      cidr: 10.0.10.0/24
      availability_zone: us-east-1a
      public: false
    - name: services-2
      cidr: 10.0.11.0/24
      availability_zone: us-east-1b
      public: false
```

## Best Practices

### 1. Use Multi-AZ Deployment

Always deploy across multiple availability zones for high availability:

```yaml
networking:
  vpc:
    cidr: 10.0.0.0/16
  subnets:
    - name: public-1
      cidr: 10.0.1.0/24
      availability_zone: us-east-1a
      public: true
    - name: public-2
      cidr: 10.0.2.0/24
      availability_zone: us-east-1b
      public: true
```

### 2. Follow Security Group Best Practices

- Use specific port ranges
- Reference security groups by name
- Document your rules with descriptions
- Follow the principle of least privilege

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
```

### 3. Use Appropriate CIDR Blocks

- Use `/16` for VPC (65,536 IPs)
- Use `/24` for subnets (256 IPs)
- Plan for growth and future subnets

### 4. Enable VPC Flow Logs

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

## Troubleshooting

### Common Issues

1. **Services can't communicate**: Check security group rules
2. **Load balancer health checks failing**: Verify security group ingress rules
3. **DNS resolution issues**: Ensure DNS support is enabled in VPC

### Debug Commands

```bash
# Check VPC configuration
aws ec2 describe-vpcs --vpc-ids vpc-12345678

# Check subnet configuration
aws ec2 describe-subnets --filters "Name=vpc-id,Values=vpc-12345678"

# Check security groups
aws ec2 describe-security-groups --group-ids sg-12345678
```

## Example: Complete Networking Setup

```yaml
# networking.yaml
networking:
  vpc:
    cidr: 10.0.0.0/16
    enable_dns_hostnames: true
    enable_dns_support: true
    flow_logs:
      enabled: true
      destination: cloudwatch
      log_group: /aws/vpc/flowlogs
      retention_days: 30
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
      cidr: 10.0.10.0/24
      availability_zone: us-east-1a
      public: false
    - name: private-2
      cidr: 10.0.11.0/24
      availability_zone: us-east-1b
      public: false

security_groups:
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
```

This configuration provides a complete, production-ready networking setup with proper security and high availability.

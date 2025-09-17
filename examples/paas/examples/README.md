# Cool Demo PaaS Examples

This directory contains example YAML configurations demonstrating various use cases for Cool Demo PaaS.

## Examples

### 1. Simple Web Application (`simple-web-app.yaml`)

A basic web application with:
- ECS Fargate service running nginx
- Application Load Balancer with HTTPS
- Auto scaling configuration
- SSL certificate management
- DNS configuration

**Use case**: Perfect for getting started with Cool Demo PaaS or hosting a simple website.

### 2. Microservices Architecture (`microservices-example.yaml`)

A microservices architecture with:
- Multiple ECS services (user, order, product services)
- API Gateway with path-based routing
- Individual auto scaling for each service
- Service-specific security groups
- Centralized logging and monitoring

**Use case**: Ideal for complex applications that need to scale different components independently.

## How to Use These Examples

1. **Copy the example** that best fits your use case
2. **Modify the configuration** to match your requirements:
   - Update domain names
   - Change image references
   - Adjust resource allocations
   - Modify security group rules
3. **Deploy your infrastructure** using Cool Demo PaaS

## Customization Tips

### Domain Names
Replace `example.com` with your actual domain:
```yaml
certificates:
  web-cert:
    domain: your-domain.com
    subject_alternative_names:
      - www.your-domain.com
```

### Docker Images
Update image references to your actual container images:
```yaml
services:
  web-app:
    image: your-registry/your-app:latest
```

### Resource Sizing
Adjust CPU and memory based on your application needs:
```yaml
services:
  web-app:
    cpu: 1024      # Increase for more compute power
    memory: 2048   # Increase for memory-intensive applications
```

### Scaling Configuration
Modify auto scaling settings:
```yaml
services:
  web-app:
    min_capacity: 1      # Minimum number of instances
    max_capacity: 20     # Maximum number of instances
    desired_capacity: 5  # Initial number of instances
```

## Next Steps

1. **Read the documentation** - Check out our [DSL Reference](/dsl/ec2) for detailed configuration options
2. **Follow best practices** - See our [Best Practices](/guidelines/best-practices) guide
3. **Learn about networking** - Review our [Networking Guidelines](/guidelines/networking)
4. **Explore how-tos** - Follow our [How-tos](/guidelines/how-tos) for step-by-step instructions

## Need Help?

- Check our [documentation](/intro) for comprehensive guides
- Review the [DSL reference](/dsl/ec2) for configuration options
- Follow our [best practices](/guidelines/best-practices) for production deployments

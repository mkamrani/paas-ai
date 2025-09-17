# Welcome to Cool Demo PaaS

Cool Demo PaaS is a simplified Platform as a Service that allows you to define AWS infrastructure using a simple YAML-based Domain Specific Language (DSL). Our goal is to make cloud infrastructure accessible and easy to understand.

## What is Cool Demo PaaS?

Cool Demo PaaS provides a simplified way to define and deploy AWS infrastructure using YAML configuration files. Instead of dealing with complex CloudFormation templates or Terraform configurations, you can define your infrastructure using our intuitive DSL.

## Supported AWS Services

Our PaaS currently supports the following AWS services:

- **EC2**: Virtual machines with simple instance type selection
- **ECS on Fargate**: Containerized applications with managed compute
- **Application Load Balancer (ALB)**: HTTP/HTTPS load balancing
- **Route53**: DNS management and domain routing
- **AWS Certificate Manager (ACM)**: SSL/TLS certificate management

## Key Features

- **Simple YAML Syntax**: Easy-to-read configuration files
- **Multi-file Support**: Organize your infrastructure across multiple YAML files
- **Sensible Defaults**: Pre-configured networking and security settings
- **Best Practices**: Built-in recommendations for production deployments

## Quick Start

1. **Define your infrastructure** using our YAML DSL
2. **Organize your configuration** across multiple files
3. **Follow our guidelines** for best practices
4. **Deploy with confidence** knowing your infrastructure follows AWS best practices

## Getting Started

Ready to start building? Check out our [DSL Reference](/dsl/ec2) to learn about defining your first infrastructure components.

## Example

Here's a simple example of what your infrastructure might look like:

```yaml
# infrastructure.yaml
project:
  name: my-web-app
  environment: production

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

services:
  web:
    type: ecs
    cpu: 256
    memory: 512
    image: nginx:latest
    port: 80
    desired_count: 2
```

This example creates a simple web application running on ECS Fargate with basic networking setup.

## Next Steps

- Explore the [DSL Reference](/dsl/ec2) to understand available services
- Read our [Best Practices](/guidelines/best-practices) for production deployments
- Check out [Networking Guidelines](/guidelines/networking) for infrastructure design

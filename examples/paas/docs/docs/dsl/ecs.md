# ECS - Elastic Container Service

ECS provides a serverless container platform that eliminates the need to manage servers. Our PaaS automatically uses Fargate for all ECS services, making it easy to deploy containerized applications.

## Basic Configuration

```yaml
services:
  web-app:
    type: ecs
    cpu: 256
    memory: 512
    image: nginx:latest
    port: 80
    desired_count: 2
    environment:
      - name: NODE_ENV
        value: production
```

## Configuration Options

### Compute Resources

| CPU (vCPU) | Memory (GB) | Use Case |
|------------|-------------|----------|
| 256 | 0.5, 1, 2 | Light workloads, development |
| 512 | 1, 2, 4 | Small applications |
| 1024 | 2, 4, 8 | Medium applications |
| 2048 | 4, 8, 16 | Large applications |
| 4096 | 8, 16, 30 | High-performance applications |

### Container Configuration

```yaml
services:
  api:
    type: ecs
    cpu: 512
    memory: 1024
    image: my-app:latest
    port: 3000
    desired_count: 3
    environment:
      - name: DATABASE_URL
        value: "postgresql://user:pass@db:5432/mydb"
      - name: REDIS_URL
        value: "redis://cache:6379"
    secrets:
      - name: API_KEY
        value: "my-api-key-secret"
      - name: JWT_SECRET
        value: "jwt-secret-name"
    health_check:
      command: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 30
      timeout: 5
      retries: 3
```

### Load Balancer Integration

```yaml
services:
  web-app:
    type: ecs
    cpu: 256
    memory: 512
    image: nginx:latest
    port: 80
    desired_count: 2
    health_check_path: /health
```

## Networking

ECS services automatically use the PaaS networking configuration. You can specify which subnets to use:

```yaml
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
      - private-2
    security_groups:
      - web-sg
```

### Default Networking Behavior

If you don't specify subnets or security groups, the PaaS will:
- Use private subnets by default
- Create appropriate security groups automatically
- Configure proper ingress/egress rules

## Auto Scaling

```yaml
services:
  api:
    type: ecs
    cpu: 512
    memory: 1024
    image: my-api:latest
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
```

## Multi-Container Services

```yaml
services:
  web-app:
    type: ecs
    cpu: 512
    memory: 1024
    containers:
      - name: nginx
        image: nginx:alpine
        port: 80
        cpu: 128
        memory: 256
      - name: app
        image: my-app:latest
        port: 3000
        cpu: 384
        memory: 768
        environment:
          - name: NODE_ENV
            value: production
    desired_count: 2
```

## Service Discovery

```yaml
services:
  api:
    type: ecs
    cpu: 256
    memory: 512
    image: my-api:latest
    port: 3000
    desired_count: 2
    service_discovery:
      namespace: my-app.local
      service_name: api
```

## Secrets Management

The PaaS automatically handles secrets management by mapping to AWS Secrets Manager in the correct account:

```yaml
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
        value: "jwt-signing-secret"
```

The PaaS will:
- Automatically create secrets in AWS Secrets Manager
- Map the secret names to the correct ARNs
- Handle secret rotation and updates
- Ensure proper IAM permissions

## Best Practices

1. **Right-size your containers** - Use appropriate CPU and memory allocations
2. **Use health checks** - Implement proper health check endpoints
3. **Enable auto scaling** - Configure scaling policies for variable workloads
4. **Use secrets management** - Store sensitive data using the simplified secrets syntax
5. **Implement logging** - Use CloudWatch Logs for container logs
6. **Use service discovery** - For microservices communication

## Example: Complete Web Application

```yaml
# web-app.yaml
services:
  web-app:
    type: ecs
    cpu: 512
    memory: 1024
    image: my-web-app:latest
    port: 3000
    min_capacity: 2
    max_capacity: 8
    desired_count: 3
    subnets:
      - private-1
      - private-2
    security_groups:
      - web-sg
    environment:
      - name: NODE_ENV
        value: production
      - name: DATABASE_URL
        value: "postgresql://user:pass@db:5432/mydb"
    secrets:
      - name: JWT_SECRET
        value: "jwt-secret-name"
      - name: STRIPE_KEY
        value: "stripe-secret-key"
    health_check:
      command: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 30
      timeout: 5
      retries: 3
    scaling_policies:
      - metric: cpu_utilization
        target_value: 70
        scale_out_cooldown: 300
        scale_in_cooldown: 300
```

This configuration creates a scalable web application with proper health checks, secrets management, and auto scaling.

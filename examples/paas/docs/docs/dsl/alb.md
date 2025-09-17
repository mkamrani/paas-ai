# ALB - Application Load Balancer

Application Load Balancer provides HTTP/HTTPS load balancing with advanced routing capabilities. Our PaaS simplifies ALB configuration by automatically managing target groups and health checks.

## Basic Configuration

```yaml
load_balancers:
  web-alb:
    type: alb
    scheme: internet-facing
    listeners:
      - port: 80
        protocol: HTTP
        default_action:
          type: forward
          service: web-app
      - port: 443
        protocol: HTTPS
        certificate: my-domain-cert
        default_action:
          type: forward
          service: web-app
```

## Configuration Options

### Listener Configuration

```yaml
load_balancers:
  api-alb:
    type: alb
    scheme: internal
    listeners:
      - port: 80
        protocol: HTTP
        rules:
          - condition:
              path_pattern: "/api/*"
            action:
              type: forward
              service: api-service
          - condition:
              host_header: "api.example.com"
            action:
              type: forward
              service: api-service
          - action:
              type: forward
              service: default-service
```

### SSL/TLS Configuration

```yaml
load_balancers:
  secure-alb:
    type: alb
    scheme: internet-facing
    listeners:
      - port: 443
        protocol: HTTPS
        certificate: my-domain-cert
        ssl_policy: ELBSecurityPolicy-TLS-1-2-2017-01
        default_action:
          type: forward
          service: web-app
      - port: 80
        protocol: HTTP
        default_action:
          type: redirect
          redirect_config:
            protocol: HTTPS
            port: 443
            status_code: HTTP_301
```

## Advanced Routing

### Path-based Routing

```yaml
load_balancers:
  microservices-alb:
    type: alb
    scheme: internet-facing
    listeners:
      - port: 80
        protocol: HTTP
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
          - action:
              type: forward
              service: default-service
```

### Host-based Routing

```yaml
load_balancers:
  multi-domain-alb:
    type: alb
    scheme: internet-facing
    listeners:
      - port: 443
        protocol: HTTPS
        certificate: wildcard-cert
        rules:
          - condition:
              host_header: "api.example.com"
            action:
              type: forward
              service: api-service
          - condition:
              host_header: "admin.example.com"
            action:
              type: forward
              service: admin-service
          - condition:
              host_header: "www.example.com"
            action:
              type: forward
              service: web-service
```

### Weighted Routing

```yaml
load_balancers:
  canary-alb:
    type: alb
    scheme: internet-facing
    listeners:
      - port: 80
        protocol: HTTP
        rules:
          - condition:
              path_pattern: "/api/*"
            action:
              type: forward
              services:
                - service: api-v1
                  weight: 90
                - service: api-v2
                  weight: 10
```

## Health Checks

The PaaS automatically configures health checks based on your service configuration:

```yaml
services:
  api-service:
    type: ecs
    cpu: 512
    memory: 1024
    image: my-api:latest
    port: 3000
    health_check:
      path: /health
      interval: 30
      timeout: 5
      retries: 3
```

The load balancer will automatically use this health check configuration for routing decisions.

## Best Practices

1. **Use HTTPS** - Always redirect HTTP to HTTPS for security
2. **Implement health checks** - Use meaningful health check endpoints in your services
3. **Use appropriate SSL policies** - Choose the right SSL policy for your needs
4. **Configure proper timeouts** - Set appropriate idle and request timeouts
5. **Use path-based routing** - For microservices architectures
6. **Enable access logs** - For monitoring and debugging

## Example: Complete Load Balancer Setup

```yaml
# load-balancer.yaml
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
        certificate: example-com-cert
        ssl_policy: ELBSecurityPolicy-TLS-1-2-2017-01
        rules:
          - condition:
              path_pattern: "/api/*"
            action:
              type: forward
              service: api-service
          - condition:
              path_pattern: "/admin/*"
            action:
              type: forward
              service: admin-service
          - action:
              type: forward
              service: web-service

# Services automatically get proper health checks
services:
  web-service:
    type: ecs
    cpu: 512
    memory: 1024
    image: web-app:latest
    port: 80
    health_check:
      path: /health
      interval: 30
      timeout: 5
      retries: 3

  api-service:
    type: ecs
    cpu: 512
    memory: 1024
    image: api-app:latest
    port: 3000
    health_check:
      path: /health
      interval: 30
      timeout: 5
      retries: 3

  admin-service:
    type: ecs
    cpu: 256
    memory: 512
    image: admin-app:latest
    port: 8080
    health_check:
      path: /health
      interval: 30
      timeout: 5
      retries: 3
```

This configuration creates a comprehensive load balancer setup with HTTPS redirect, path-based routing, and automatic health check management.

# Route53 - DNS Management

Route53 provides DNS services for your domains. Our PaaS simplifies DNS configuration for common use cases like web applications and API endpoints.

## Basic Configuration

```yaml
dns:
  zones:
    - domain: example.com
      records:
        - name: www
          type: A
          alias: true
          target: web-alb
        - name: api
          type: A
          alias: true
          target: api-alb
```

## Configuration Options

### Record Types

#### A Records (IPv4)

```yaml
dns:
  zones:
    - domain: example.com
      records:
        - name: www
          type: A
          alias: true
          target: web-alb
        - name: app
          type: A
          value: 192.168.1.100
          ttl: 300
```

#### AAAA Records (IPv6)

```yaml
dns:
  zones:
    - domain: example.com
      records:
        - name: www
          type: AAAA
          alias: true
          target: web-alb
```

#### CNAME Records

```yaml
dns:
  zones:
    - domain: example.com
      records:
        - name: blog
          type: CNAME
          value: blog.example.com
          ttl: 300
```

#### MX Records (Mail)

```yaml
dns:
  zones:
    - domain: example.com
      records:
        - name: ""
          type: MX
          value: "10 mail.example.com"
          ttl: 300
        - name: ""
          type: MX
          value: "20 mail2.example.com"
          ttl: 300
```

#### TXT Records

```yaml
dns:
  zones:
    - domain: example.com
      records:
        - name: ""
          type: TXT
          value: "v=spf1 include:_spf.google.com ~all"
          ttl: 300
        - name: _verification
          type: TXT
          value: "google-site-verification=abc123"
          ttl: 300
```

### Alias Records

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

### Subdomain Configuration

```yaml
dns:
  zones:
    - domain: example.com
      records:
        - name: www
          type: A
          alias: true
          target: web-alb
        - name: api
          type: A
          alias: true
          target: api-alb
        - name: admin
          type: A
          alias: true
          target: admin-alb
        - name: staging
          type: A
          alias: true
          target: staging-alb
```

### Wildcard Records

```yaml
dns:
  zones:
    - domain: example.com
      records:
        - name: "*"
          type: A
          alias: true
          target: web-alb
        - name: "*.api"
          type: A
          alias: true
          target: api-alb
```

## Health Checks

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
          health_check:
            path: /health
            port: 80
            protocol: HTTP
            failure_threshold: 3
            request_interval: 30
```

## Failover Configuration

```yaml
dns:
  zones:
    - domain: example.com
      records:
        - name: www
          type: A
          alias: true
          target: primary-alb
          evaluate_target_health: true
          failover: PRIMARY
        - name: www
          type: A
          alias: true
          target: secondary-alb
          evaluate_target_health: true
          failover: SECONDARY
```

## Geographic Routing

```yaml
dns:
  zones:
    - domain: example.com
      records:
        - name: www
          type: A
          alias: true
          target: us-alb
          evaluate_target_health: true
          geolocation:
            continent: NORTH_AMERICA
        - name: www
          type: A
          alias: true
          target: eu-alb
          evaluate_target_health: true
          geolocation:
            continent: EUROPE
```

## Best Practices

1. **Use alias records** - For AWS resources, use alias records instead of CNAME
2. **Enable health checks** - Use `evaluate_target_health: true` for better reliability
3. **Set appropriate TTLs** - Balance between performance and flexibility
4. **Use failover routing** - For high availability applications
5. **Implement geographic routing** - For global applications
6. **Use wildcard records carefully** - Only when necessary for subdomain management

## Example: Complete DNS Setup

```yaml
# dns.yaml
dns:
  zones:
    - domain: example.com
      records:
        # Main website
        - name: www
          type: A
          alias: true
          target: web-alb
          evaluate_target_health: true
        - name: ""
          type: A
          alias: true
          target: web-alb
          evaluate_target_health: true
        
        # API endpoints
        - name: api
          type: A
          alias: true
          target: api-alb
          evaluate_target_health: true
        - name: "*.api"
          type: A
          alias: true
          target: api-alb
          evaluate_target_health: true
        
        # Admin interface
        - name: admin
          type: A
          alias: true
          target: admin-alb
          evaluate_target_health: true
        
        # Mail records
        - name: ""
          type: MX
          value: "10 mail.example.com"
          ttl: 300
        - name: mail
          type: A
          value: 192.168.1.100
          ttl: 300
        
        # SPF record
        - name: ""
          type: TXT
          value: "v=spf1 include:_spf.google.com ~all"
          ttl: 300
        
        # CNAME for blog
        - name: blog
          type: CNAME
          value: blog.example.com
          ttl: 300
```

This configuration sets up a comprehensive DNS structure for a web application with API, admin interface, and email services.

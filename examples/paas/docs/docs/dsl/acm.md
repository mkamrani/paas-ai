# ACM - AWS Certificate Manager

AWS Certificate Manager provides SSL/TLS certificates for your applications. Our PaaS simplifies certificate management and automatic renewal.

## Basic Configuration

```yaml
certificates:
  web-cert:
    domain: example.com
    subject_alternative_names:
      - www.example.com
      - api.example.com
    validation_method: DNS
```

## Configuration Options

### Domain Validation

#### DNS Validation (Recommended)

```yaml
certificates:
  web-cert:
    domain: example.com
    subject_alternative_names:
      - www.example.com
      - api.example.com
      - admin.example.com
    validation_method: DNS
    auto_validation: true
```

#### Email Validation

```yaml
certificates:
  web-cert:
    domain: example.com
    subject_alternative_names:
      - www.example.com
    validation_method: EMAIL
    validation_emails:
      - admin@example.com
      - administrator@example.com
```

### Wildcard Certificates

```yaml
certificates:
  wildcard-cert:
    domain: "*.example.com"
    validation_method: DNS
    auto_validation: true
```

### Multi-Domain Certificates

```yaml
certificates:
  multi-domain-cert:
    domain: example.com
    subject_alternative_names:
      - www.example.com
      - api.example.com
      - admin.example.com
      - staging.example.com
      - "*.api.example.com"
    validation_method: DNS
    auto_validation: true
```

## Certificate Usage

### Load Balancer Integration

```yaml
certificates:
  web-cert:
    domain: example.com
    subject_alternative_names:
      - www.example.com
      - api.example.com
    validation_method: DNS
    auto_validation: true

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
```

### CloudFront Integration

```yaml
certificates:
  cloudfront-cert:
    domain: example.com
    subject_alternative_names:
      - www.example.com
    validation_method: DNS
    auto_validation: true
    region: us-east-1  # Required for CloudFront

distributions:
  web-cdn:
    type: cloudfront
    certificate: cloudfront-cert
    origins:
      - domain: web-alb.example.com
        path_pattern: "/*"
```

## Certificate Renewal

```yaml
certificates:
  web-cert:
    domain: example.com
    subject_alternative_names:
      - www.example.com
    validation_method: DNS
    auto_validation: true
    auto_renewal: true
    renewal_notification:
      - email: admin@example.com
      - email: devops@example.com
```

## Certificate Monitoring

```yaml
certificates:
  web-cert:
    domain: example.com
    subject_alternative_names:
      - www.example.com
    validation_method: DNS
    auto_validation: true
    monitoring:
      expiration_alert_days: 30
      notification_emails:
        - admin@example.com
        - devops@example.com
```

## Best Practices

1. **Use DNS validation** - More reliable than email validation
2. **Enable auto-validation** - Automatically handle DNS record creation
3. **Use wildcard certificates** - For subdomain management
4. **Monitor expiration** - Set up alerts for certificate expiration
5. **Use appropriate regions** - Use us-east-1 for CloudFront certificates
6. **Plan for renewal** - Certificates auto-renew, but monitor the process

## Example: Complete Certificate Setup

```yaml
# certificates.yaml
certificates:
  # Main website certificate
  web-cert:
    domain: example.com
    subject_alternative_names:
      - www.example.com
      - api.example.com
      - admin.example.com
    validation_method: DNS
    auto_validation: true
    auto_renewal: true
    monitoring:
      expiration_alert_days: 30
      notification_emails:
        - admin@example.com
        - devops@example.com

  # Wildcard certificate for subdomains
  wildcard-cert:
    domain: "*.example.com"
    validation_method: DNS
    auto_validation: true
    auto_renewal: true
    monitoring:
      expiration_alert_days: 30
      notification_emails:
        - admin@example.com

  # Staging environment certificate
  staging-cert:
    domain: staging.example.com
    subject_alternative_names:
      - "*.staging.example.com"
    validation_method: DNS
    auto_validation: true
    auto_renewal: true

  # CloudFront certificate (must be in us-east-1)
  cloudfront-cert:
    domain: example.com
    subject_alternative_names:
      - www.example.com
    validation_method: DNS
    auto_validation: true
    auto_renewal: true
    region: us-east-1
    monitoring:
      expiration_alert_days: 30
      notification_emails:
        - admin@example.com
```

This configuration sets up comprehensive SSL/TLS certificate management for a multi-environment application with proper monitoring and auto-renewal.

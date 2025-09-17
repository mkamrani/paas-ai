# RDS - Relational Database Service

RDS provides managed relational databases in the cloud. Our PaaS simplifies RDS configuration with sensible defaults and easy setup.

## Basic Configuration

```yaml
services:
  database:
    type: rds
    engine: postgres
    engine_version: "13.7"
    instance_class: db.t3.micro
    allocated_storage: 20
    multi_az: true
    backup_retention_period: 7
```

## Configuration Options

### Database Engines

| Engine | Versions | Use Case |
|--------|----------|----------|
| postgres | 11.x, 12.x, 13.x, 14.x | General purpose, ACID-compliant |
| mysql | 5.7, 8.0 | Web applications, e-commerce |
| mariadb | 10.3, 10.4, 10.5, 10.6 | Drop-in replacement for MySQL |
| aurora-postgresql | 11.x, 12.x, 13.x | High performance PostgreSQL |
| aurora-mysql | 5.7, 8.0 | High performance MySQL |
| sqlserver-ee | 2019, 2022 | Enterprise applications |
| sqlserver-se | 2019, 2022 | Medium-sized business applications |
| sqlserver-ex | 2019, 2022 | Small applications, development |
| oracle-ee | 19c, 21c | Enterprise applications |
| oracle-se2 | 19c, 21c | Medium-sized business applications |

### Instance Classes

| Category | Classes | Use Case |
|----------|---------|----------|
| Burstable | db.t3.micro, db.t3.small, db.t3.medium, db.t3.large | Development, testing, small applications |
| Standard | db.m5.large, db.m5.xlarge, db.m5.2xlarge, db.m5.4xlarge | Production applications with balanced performance |
| Memory Optimized | db.r5.large, db.r5.xlarge, db.r5.2xlarge, db.r5.4xlarge | Memory-intensive applications, caching |
| Performance Optimized | db.c5.large, db.c5.xlarge, db.c5.2xlarge, db.c5.4xlarge | Compute-intensive applications |

### Storage Configuration

```yaml
services:
  database:
    type: rds
    engine: postgres
    engine_version: "13.7"
    instance_class: db.t3.micro
    allocated_storage: 20
    max_allocated_storage: 100
    backup_retention_period: 7
    backup_window: "03:00-04:00"
    maintenance_window: "sun:04:00-sun:05:00"
```

### Multi-AZ Deployment

```yaml
services:
  database:
    type: rds
    engine: postgres
    engine_version: "13.7"
    instance_class: db.t3.micro
    allocated_storage: 20
    multi_az: true
    backup_retention_period: 7
```

### Network Configuration

```yaml
services:
  database:
    type: rds
    engine: postgres
    engine_version: "13.7"
    instance_class: db.t3.micro
    allocated_storage: 20
    subnets:
      - database-1
      - database-2
    security_groups:
      - database-sg
```

### Database Parameters

```yaml
services:
  postgres-db:
    type: rds
    engine: postgres
    engine_version: "13.7"
    instance_class: db.t3.micro
    allocated_storage: 20
    parameters:
      - name: shared_buffers
        value: "256MB"
      - name: max_connections
        value: "100"
      - name: shared_preload_libraries
        value: "pg_stat_statements"
```

### Credentials Management

```yaml
services:
  database:
    type: rds
    engine: postgres
    engine_version: "13.7"
    instance_class: db.t3.micro
    allocated_storage: 20
    credentials:
      username: "db_user"
      password_secret: "database-password"
```

The PaaS will:
- Automatically create the secret in AWS Secrets Manager
- Securely store and rotate the password
- Ensure proper IAM permissions

## Advanced Configuration

### Read Replicas

```yaml
services:
  primary-db:
    type: rds
    engine: postgres
    engine_version: "13.7"
    instance_class: db.m5.large
    allocated_storage: 100
    multi_az: true
  
  read-replica:
    type: rds
    replica_of: primary-db
    instance_class: db.r5.large
    region: us-west-2  # Cross-region replica
```

### Enhanced Monitoring

```yaml
services:
  database:
    type: rds
    engine: postgres
    engine_version: "13.7"
    instance_class: db.t3.micro
    allocated_storage: 20
    monitoring:
      interval: 30  # seconds
      role: "rds-monitoring-role"
      logs_exports:
        - postgresql
        - upgrade
```

### Performance Insights

```yaml
services:
  database:
    type: rds
    engine: postgres
    engine_version: "13.7"
    instance_class: db.t3.micro
    allocated_storage: 20
    performance_insights:
      enabled: true
      retention_period: 7  # days
```

## Best Practices

1. **Use appropriate instance classes** for your workload
2. **Enable Multi-AZ** for production databases
3. **Configure automated backups** with appropriate retention
4. **Use parameter groups** to optimize database performance
5. **Enable encryption** for sensitive data
6. **Use read replicas** for read-heavy workloads
7. **Monitor performance** with Enhanced Monitoring and Performance Insights

## Example: Complete Database Setup

```yaml
# database.yaml
services:
  postgres-db:
    type: rds
    engine: postgres
    engine_version: "13.7"
    instance_class: db.t3.medium
    allocated_storage: 50
    multi_az: true
    subnets:
      - database-1
      - database-2
    security_groups:
      - database-sg
    backup_retention_period: 7
    backup_window: "03:00-04:00"
    maintenance_window: "sun:04:00-sun:05:00"
    parameters:
      - name: shared_buffers
        value: "256MB"
      - name: max_connections
        value: "200"
    monitoring:
      interval: 30
      logs_exports:
        - postgresql
        - upgrade
    performance_insights:
      enabled: true
      retention_period: 7
    credentials:
      username: "app_user"
      password_secret: "database-password"
    tags:
      Environment: production
      Service: user-database
```

This configuration creates a production-ready PostgreSQL database with proper security, backups, and monitoring.

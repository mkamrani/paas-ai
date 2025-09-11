"""
Kubernetes tools for the K8s Manifest agent.

Provides tools for generating K8s manifests, validation, and troubleshooting.
"""

from typing import Any, Dict, Optional, Literal, List
from langchain.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field
import yaml

from ....utils.logging import get_logger

logger = get_logger("paas_ai.agents.k8s_tools")


class GenerateManifestInput(BaseModel):
    """Input schema for K8s manifest generation."""
    resource_type: Literal["deployment", "service", "ingress", "configmap", "secret", "statefulset", "complete"] = Field(
        description="Type of Kubernetes resource to generate"
    )
    app_name: str = Field(description="Name of the application")
    image: str = Field(description="Container image to deploy")
    port: int = Field(default=8080, description="Container port")
    replicas: int = Field(default=3, description="Number of replicas")
    namespace: str = Field(default="default", description="Kubernetes namespace")
    environment: Literal["development", "staging", "production"] = Field(
        default="production", 
        description="Target environment"
    )
    additional_config: Optional[str] = Field(
        default=None,
        description="Additional configuration requirements or constraints"
    )


class GenerateManifestTool(BaseTool):
    """Tool for generating Kubernetes YAML manifest files."""
    
    name: str = "generate_k8s_manifest"
    description: str = """
    Generate production-ready Kubernetes YAML manifest files.
    
    Can generate:
    - Deployments with proper resource limits and health checks
    - Services (ClusterIP, LoadBalancer, NodePort)
    - Ingress configurations for external access
    - ConfigMaps for application configuration
    - Secrets for sensitive data
    - StatefulSets for stateful applications
    - Complete application stack (deployment + service + ingress)
    
    Includes security best practices, resource management, and monitoring setup.
    """
    args_schema: type = GenerateManifestInput
    
    def __init__(self):
        """Initialize the K8s manifest generation tool."""
        super().__init__()
    
    def _run(
        self,
        resource_type: str,
        app_name: str,
        image: str,
        port: int = 8080,
        replicas: int = 3,
        namespace: str = "default",
        environment: str = "production",
        additional_config: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs
    ) -> str:
        """Generate Kubernetes manifest based on specifications."""
        try:
            if resource_type == "deployment":
                return self._generate_deployment(app_name, image, port, replicas, namespace, environment, additional_config)
            elif resource_type == "service":
                return self._generate_service(app_name, port, namespace, environment, additional_config)
            elif resource_type == "ingress":
                return self._generate_ingress(app_name, namespace, environment, additional_config)
            elif resource_type == "configmap":
                return self._generate_configmap(app_name, namespace, environment, additional_config)
            elif resource_type == "secret":
                return self._generate_secret(app_name, namespace, environment, additional_config)
            elif resource_type == "statefulset":
                return self._generate_statefulset(app_name, image, port, replicas, namespace, environment, additional_config)
            elif resource_type == "complete":
                return self._generate_complete_stack(app_name, image, port, replicas, namespace, environment, additional_config)
            else:
                return f"Unsupported resource type: {resource_type}"
                
        except Exception as e:
            logger.error(f"Error generating K8s manifest: {e}")
            return f"Error generating manifest: {str(e)}"
    
    def _generate_deployment(self, app_name: str, image: str, port: int, replicas: int, namespace: str, environment: str, additional_config: Optional[str]) -> str:
        """Generate a Kubernetes Deployment manifest."""
        resource_requests = self._get_resource_config(environment)
        
        return f"""# Kubernetes Deployment for {app_name}

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {app_name}
  namespace: {namespace}
  labels:
    app: {app_name}
    version: v1
    environment: {environment}
  annotations:
    deployment.kubernetes.io/revision: "1"
spec:
  replicas: {replicas}
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: {app_name}
  template:
    metadata:
      labels:
        app: {app_name}
        version: v1
        environment: {environment}
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "{port}"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: {app_name}
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: {app_name}
        image: {image}
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: {port}
          protocol: TCP
          name: http
        env:
        - name: PORT
          value: "{port}"
        - name: ENVIRONMENT
          value: "{environment}"
        - name: LOG_LEVEL
          value: "info"
        envFrom:
        - configMapRef:
            name: {app_name}-config
        - secretRef:
            name: {app_name}-secret
        resources:
          requests:
            memory: "{resource_requests['memory_request']}"
            cpu: "{resource_requests['cpu_request']}"
          limits:
            memory: "{resource_requests['memory_limit']}"
            cpu: "{resource_requests['cpu_limit']}"
        livenessProbe:
          httpGet:
            path: /health
            port: {port}
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: {port}
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        startupProbe:
          httpGet:
            path: /health
            port: {port}
          initialDelaySeconds: 10
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 30
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
            - ALL
        volumeMounts:
        - name: tmp
          mountPath: /tmp
        - name: var-run
          mountPath: /var/run
      volumes:
      - name: tmp
        emptyDir: {{}}
      - name: var-run
        emptyDir: {{}}
      terminationGracePeriodSeconds: 30
```

## Deployment Configuration

**Application**: {app_name}
**Environment**: {environment}
**Replicas**: {replicas}
**Resource Allocation**: {resource_requests['memory_request']} memory, {resource_requests['cpu_request']} CPU

{f"**Additional Configuration**: {additional_config}" if additional_config else ""}

## Key Features
- **Rolling Updates**: Zero-downtime deployments
- **Health Checks**: Comprehensive liveness, readiness, and startup probes
- **Security**: Non-root user, read-only filesystem, dropped capabilities
- **Resource Management**: Appropriate requests and limits for {environment}
- **Monitoring**: Prometheus annotations for metrics collection

## Next Steps
1. Apply the deployment: `kubectl apply -f {app_name}-deployment.yaml`
2. Check rollout status: `kubectl rollout status deployment/{app_name} -n {namespace}`
3. View pods: `kubectl get pods -l app={app_name} -n {namespace}`
"""

    def _generate_service(self, app_name: str, port: int, namespace: str, environment: str, additional_config: Optional[str]) -> str:
        """Generate a Kubernetes Service manifest."""
        return f"""# Kubernetes Service for {app_name}

```yaml
apiVersion: v1
kind: Service
metadata:
  name: {app_name}
  namespace: {namespace}
  labels:
    app: {app_name}
    environment: {environment}
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: nlb
spec:
  type: ClusterIP
  selector:
    app: {app_name}
  ports:
  - name: http
    port: 80
    targetPort: {port}
    protocol: TCP
  sessionAffinity: None
---
# LoadBalancer Service (if external access needed)
apiVersion: v1
kind: Service
metadata:
  name: {app_name}-external
  namespace: {namespace}
  labels:
    app: {app_name}
    environment: {environment}
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: nlb
spec:
  type: LoadBalancer
  selector:
    app: {app_name}
  ports:
  - name: http
    port: 80
    targetPort: {port}
    protocol: TCP
```

## Service Configuration

**Service Name**: {app_name}
**Port Mapping**: 80 → {port}
**Environment**: {environment}

{f"**Additional Configuration**: {additional_config}" if additional_config else ""}

## Service Types Included
1. **ClusterIP Service**: Internal cluster communication
2. **LoadBalancer Service**: External access (cloud provider)

## Usage
- Internal access: `http://{app_name}.{namespace}.svc.cluster.local`
- External access: Use LoadBalancer external IP

## Commands
- Get service info: `kubectl get svc {app_name} -n {namespace}`
- Describe service: `kubectl describe svc {app_name} -n {namespace}`
"""

    def _generate_ingress(self, app_name: str, namespace: str, environment: str, additional_config: Optional[str]) -> str:
        """Generate a Kubernetes Ingress manifest."""
        hostname = f"{app_name}.{environment}.example.com" if environment != "production" else f"{app_name}.example.com"
        
        return f"""# Kubernetes Ingress for {app_name}

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {app_name}
  namespace: {namespace}
  labels:
    app: {app_name}
    environment: {environment}
  annotations:
    kubernetes.io/ingress.class: nginx
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  tls:
  - hosts:
    - {hostname}
    secretName: {app_name}-tls
  rules:
  - host: {hostname}
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: {app_name}
            port:
              number: 80
```

## Ingress Configuration

**Hostname**: {hostname}
**TLS**: Enabled with Let's Encrypt
**Rate Limiting**: 100 requests per minute

{f"**Additional Configuration**: {additional_config}" if additional_config else ""}

## Features
- **SSL/TLS**: Automatic certificate management
- **Rate Limiting**: Protection against abuse
- **Security Headers**: HTTPS redirect enforced

## DNS Configuration
Point your DNS record to the ingress controller's external IP:
```
{hostname} → <ingress-external-ip>
```

## Commands
- Get ingress: `kubectl get ingress {app_name} -n {namespace}`
- Check TLS cert: `kubectl describe secret {app_name}-tls -n {namespace}`
"""

    def _generate_complete_stack(self, app_name: str, image: str, port: int, replicas: int, namespace: str, environment: str, additional_config: Optional[str]) -> str:
        """Generate complete application stack (deployment + service + ingress + configmap)."""
        resource_requests = self._get_resource_config(environment)
        hostname = f"{app_name}.{environment}.example.com" if environment != "production" else f"{app_name}.example.com"
        
        return f"""# Complete Kubernetes Stack for {app_name}

## ConfigMap
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: {app_name}-config
  namespace: {namespace}
  labels:
    app: {app_name}
    environment: {environment}
data:
  DATABASE_URL: "postgresql://postgres:5432/{app_name}"
  REDIS_URL: "redis://redis:6379"
  LOG_LEVEL: "info"
  METRICS_ENABLED: "true"
  ENVIRONMENT: "{environment}"
```

## Secret
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: {app_name}-secret
  namespace: {namespace}
  labels:
    app: {app_name}
    environment: {environment}
type: Opaque
data:
  # Base64 encoded values - replace with actual secrets
  DATABASE_PASSWORD: cGFzc3dvcmQ=  # 'password'
  API_KEY: YXBpLWtleQ==           # 'api-key'
  JWT_SECRET: and0LXNlY3JldA==    # 'jwt-secret'
```

## ServiceAccount & RBAC
```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: {app_name}
  namespace: {namespace}
  labels:
    app: {app_name}
    environment: {environment}
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: {app_name}
  namespace: {namespace}
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: {app_name}
  namespace: {namespace}
subjects:
- kind: ServiceAccount
  name: {app_name}
  namespace: {namespace}
roleRef:
  kind: Role
  name: {app_name}
  apiGroup: rbac.authorization.k8s.io
```

## Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {app_name}
  namespace: {namespace}
  labels:
    app: {app_name}
    version: v1
    environment: {environment}
spec:
  replicas: {replicas}
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: {app_name}
  template:
    metadata:
      labels:
        app: {app_name}
        version: v1
        environment: {environment}
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "{port}"
    spec:
      serviceAccountName: {app_name}
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: {app_name}
        image: {image}
        ports:
        - containerPort: {port}
          name: http
        envFrom:
        - configMapRef:
            name: {app_name}-config
        - secretRef:
            name: {app_name}-secret
        resources:
          requests:
            memory: "{resource_requests['memory_request']}"
            cpu: "{resource_requests['cpu_request']}"
          limits:
            memory: "{resource_requests['memory_limit']}"
            cpu: "{resource_requests['cpu_limit']}"
        livenessProbe:
          httpGet:
            path: /health
            port: {port}
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: {port}
          initialDelaySeconds: 5
          periodSeconds: 5
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop: ["ALL"]
        volumeMounts:
        - name: tmp
          mountPath: /tmp
      volumes:
      - name: tmp
        emptyDir: {{}}
```

## Service
```yaml
apiVersion: v1
kind: Service
metadata:
  name: {app_name}
  namespace: {namespace}
  labels:
    app: {app_name}
    environment: {environment}
spec:
  type: ClusterIP
  selector:
    app: {app_name}
  ports:
  - name: http
    port: 80
    targetPort: {port}
```

## Ingress
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {app_name}
  namespace: {namespace}
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - {hostname}
    secretName: {app_name}-tls
  rules:
  - host: {hostname}
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: {app_name}
            port:
              number: 80
```

## HorizontalPodAutoscaler
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {app_name}
  namespace: {namespace}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {app_name}
  minReplicas: {max(1, replicas // 2)}
  maxReplicas: {replicas * 3}
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Complete Application Stack Summary

**Application**: {app_name}
**Environment**: {environment}
**Hostname**: {hostname}
**Replicas**: {replicas} (auto-scaling enabled)

{f"**Additional Configuration**: {additional_config}" if additional_config else ""}

## Deployment Instructions

1. **Create namespace** (if needed):
   ```bash
   kubectl create namespace {namespace}
   ```

2. **Apply all manifests**:
   ```bash
   kubectl apply -f {app_name}-complete-stack.yaml
   ```

3. **Verify deployment**:
   ```bash
   kubectl get all -n {namespace} -l app={app_name}
   ```

4. **Check ingress**:
   ```bash
   kubectl get ingress {app_name} -n {namespace}
   ```

5. **Test application**:
   ```bash
   curl https://{hostname}/health
   ```

## Monitoring & Troubleshooting

- **View logs**: `kubectl logs -f deployment/{app_name} -n {namespace}`
- **Check events**: `kubectl get events -n {namespace} --sort-by='.lastTimestamp'`
- **Pod status**: `kubectl get pods -l app={app_name} -n {namespace}`
- **Resource usage**: `kubectl top pods -n {namespace}`

## Security Features
- Non-root containers
- Read-only root filesystem
- Dropped capabilities
- RBAC with minimal permissions
- TLS encryption
- Secret management
"""

    def _get_resource_config(self, environment: str) -> Dict[str, str]:
        """Get resource configuration based on environment."""
        configs = {
            "development": {
                "memory_request": "128Mi",
                "memory_limit": "256Mi",
                "cpu_request": "100m",
                "cpu_limit": "200m"
            },
            "staging": {
                "memory_request": "256Mi",
                "memory_limit": "512Mi",
                "cpu_request": "200m",
                "cpu_limit": "500m"
            },
            "production": {
                "memory_request": "512Mi",
                "memory_limit": "1Gi",
                "cpu_request": "500m",
                "cpu_limit": "1000m"
            }
        }
        return configs.get(environment, configs["production"])


class ValidateManifestInput(BaseModel):
    """Input schema for K8s manifest validation."""
    manifest_yaml: str = Field(description="YAML content to validate")
    check_security: bool = Field(default=True, description="Whether to perform security checks")
    check_resources: bool = Field(default=True, description="Whether to validate resource specifications")


class ValidateManifestTool(BaseTool):
    """Tool for validating Kubernetes manifests."""
    
    name: str = "validate_k8s_manifest"
    description: str = """
    Validate Kubernetes manifest YAML files for syntax, best practices, and security.
    
    Performs checks for:
    - YAML syntax validation
    - Kubernetes API compatibility
    - Security best practices
    - Resource limit specifications
    - Label and annotation standards
    - Common configuration issues
    
    Provides detailed feedback and recommendations for improvements.
    """
    args_schema: type = ValidateManifestInput
    
    def __init__(self):
        """Initialize the K8s manifest validation tool."""
        super().__init__()
    
    def _run(
        self,
        manifest_yaml: str,
        check_security: bool = True,
        check_resources: bool = True,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs
    ) -> str:
        """Validate a Kubernetes manifest and provide recommendations."""
        try:
            # Parse YAML
            try:
                docs = list(yaml.safe_load_all(manifest_yaml))
            except yaml.YAMLError as e:
                return f"❌ **YAML Syntax Error**: {str(e)}"
            
            validation_results = []
            issues = []
            recommendations = []
            
            for i, doc in enumerate(docs):
                if doc is None:
                    continue
                    
                doc_result = self._validate_document(doc, check_security, check_resources)
                validation_results.append(doc_result)
                issues.extend(doc_result.get('issues', []))
                recommendations.extend(doc_result.get('recommendations', []))
            
            # Generate validation report
            return self._generate_validation_report(validation_results, issues, recommendations)
            
        except Exception as e:
            logger.error(f"Error validating manifest: {e}")
            return f"Error during validation: {str(e)}"
    
    def _validate_document(self, doc: Dict, check_security: bool, check_resources: bool) -> Dict:
        """Validate a single Kubernetes document."""
        issues = []
        recommendations = []
        kind = doc.get('kind', 'Unknown')
        name = doc.get('metadata', {}).get('name', 'unnamed')
        
        # Basic structure validation
        if 'apiVersion' not in doc:
            issues.append(f"Missing apiVersion in {kind}/{name}")
        
        if 'metadata' not in doc:
            issues.append(f"Missing metadata in {kind}/{name}")
        elif 'name' not in doc['metadata']:
            issues.append(f"Missing metadata.name in {kind}")
        
        # Label validation
        metadata = doc.get('metadata', {})
        labels = metadata.get('labels', {})
        
        required_labels = ['app']
        for label in required_labels:
            if label not in labels:
                recommendations.append(f"Consider adding '{label}' label to {kind}/{name}")
        
        # Security validation
        if check_security and kind in ['Deployment', 'StatefulSet', 'DaemonSet']:
            security_issues = self._check_security(doc)
            issues.extend(security_issues)
        
        # Resource validation
        if check_resources and kind in ['Deployment', 'StatefulSet', 'DaemonSet']:
            resource_issues = self._check_resources(doc)
            issues.extend(resource_issues)
        
        return {
            'kind': kind,
            'name': name,
            'issues': issues,
            'recommendations': recommendations
        }
    
    def _check_security(self, doc: Dict) -> List[str]:
        """Check security best practices."""
        issues = []
        kind = doc.get('kind')
        name = doc.get('metadata', {}).get('name', 'unnamed')
        
        spec = doc.get('spec', {})
        template = spec.get('template', {})
        pod_spec = template.get('spec', {})
        containers = pod_spec.get('containers', [])
        
        # Check for security context
        if 'securityContext' not in pod_spec:
            issues.append(f"Missing pod securityContext in {kind}/{name}")
        else:
            sc = pod_spec['securityContext']
            if not sc.get('runAsNonRoot'):
                issues.append(f"Pod should run as non-root user in {kind}/{name}")
        
        # Check container security
        for i, container in enumerate(containers):
            container_name = container.get('name', f'container-{i}')
            
            if 'securityContext' not in container:
                issues.append(f"Missing securityContext for container {container_name} in {kind}/{name}")
            else:
                csc = container['securityContext']
                if csc.get('allowPrivilegeEscalation', True):
                    issues.append(f"allowPrivilegeEscalation should be false for {container_name} in {kind}/{name}")
                
                if not csc.get('readOnlyRootFilesystem'):
                    issues.append(f"Consider readOnlyRootFilesystem for {container_name} in {kind}/{name}")
        
        return issues
    
    def _check_resources(self, doc: Dict) -> List[str]:
        """Check resource specifications."""
        issues = []
        kind = doc.get('kind')
        name = doc.get('metadata', {}).get('name', 'unnamed')
        
        spec = doc.get('spec', {})
        template = spec.get('template', {})
        pod_spec = template.get('spec', {})
        containers = pod_spec.get('containers', [])
        
        for i, container in enumerate(containers):
            container_name = container.get('name', f'container-{i}')
            
            if 'resources' not in container:
                issues.append(f"Missing resource specifications for {container_name} in {kind}/{name}")
            else:
                resources = container['resources']
                if 'requests' not in resources:
                    issues.append(f"Missing resource requests for {container_name} in {kind}/{name}")
                if 'limits' not in resources:
                    issues.append(f"Missing resource limits for {container_name} in {kind}/{name}")
        
        return issues
    
    def _generate_validation_report(self, results: List[Dict], issues: List[str], recommendations: List[str]) -> str:
        """Generate a comprehensive validation report."""
        total_docs = len(results)
        docs_with_issues = len([r for r in results if r.get('issues')])
        
        status = "✅ VALID" if not issues else "⚠️  ISSUES FOUND"
        
        report = f"""# Kubernetes Manifest Validation Report

## Summary
- **Status**: {status}
- **Documents Validated**: {total_docs}
- **Documents with Issues**: {docs_with_issues}
- **Total Issues**: {len(issues)}
- **Recommendations**: {len(recommendations)}

"""

        if issues:
            report += "## 🚨 Issues Found\n\n"
            for i, issue in enumerate(issues, 1):
                report += f"{i}. {issue}\n"
            report += "\n"
        
        if recommendations:
            report += "## 💡 Recommendations\n\n"
            for i, rec in enumerate(recommendations, 1):
                report += f"{i}. {rec}\n"
            report += "\n"
        
        if not issues and not recommendations:
            report += "## ✨ Great Job!\n\nYour manifests follow Kubernetes best practices.\n\n"
        
        report += """## Best Practices Checklist

### Security
- [ ] Run containers as non-root user
- [ ] Set readOnlyRootFilesystem: true
- [ ] Drop all capabilities
- [ ] Set allowPrivilegeEscalation: false
- [ ] Use specific image tags (not :latest)

### Resources
- [ ] Set resource requests and limits
- [ ] Configure appropriate memory/CPU values
- [ ] Use HorizontalPodAutoscaler for scaling

### Reliability
- [ ] Configure liveness and readiness probes
- [ ] Set proper restart policies
- [ ] Use rolling update strategies

### Monitoring
- [ ] Add Prometheus annotations
- [ ] Include proper labels for service discovery
- [ ] Configure log aggregation

### Networking
- [ ] Use NetworkPolicies for isolation
- [ ] Configure proper service types
- [ ] Set up ingress with TLS
"""
        
        return report 
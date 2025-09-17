"""Design tools for the Designer Agent to create structured specifications."""

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum


class ServiceType(str, Enum):
    """Supported service types in Cool Demo PaaS."""
    ECS = "ecs"
    EC2 = "ec2"
    RDS = "rds"


class ArchitecturePattern(str, Enum):
    """Common architecture patterns."""
    MICROSERVICES = "microservices"
    THREE_TIER = "three-tier"
    SERVERLESS = "serverless"
    MONOLITH = "monolith"


@dataclass
class ServiceRequirement:
    """Requirements for a single service."""
    name: str
    type: ServiceType
    description: str
    cpu: Optional[int] = None
    memory: Optional[int] = None
    scaling: Optional[Dict[str, Any]] = None
    environment_variables: Optional[List[Dict[str, str]]] = None
    secrets: Optional[List[str]] = None
    health_check_path: Optional[str] = None
    port: Optional[int] = None
    image: Optional[str] = None
    # RDS specific
    engine: Optional[str] = None
    instance_class: Optional[str] = None
    storage: Optional[int] = None


@dataclass
class NetworkingRequirement:
    """Networking requirements for the infrastructure."""
    pattern: ArchitecturePattern
    vpc_cidr: str = "10.0.0.0/16"
    multi_az: bool = True
    public_subnets: bool = True
    private_subnets: bool = True
    database_subnets: bool = False


@dataclass
class LoadBalancingRequirement:
    """Load balancing requirements."""
    type: str = "alb"
    scheme: str = "internet-facing"
    routing_type: str = "path-based"  # path-based, host-based, simple
    routes: Optional[List[Dict[str, str]]] = None
    https_redirect: bool = True


@dataclass
class SecurityRequirement:
    """Security requirements for the infrastructure."""
    https_required: bool = True
    certificate_domain: Optional[str] = None
    additional_domains: Optional[List[str]] = None
    custom_security_groups: Optional[List[Dict[str, Any]]] = None


@dataclass
class ScalingRequirement:
    """Auto-scaling requirements."""
    auto_scaling: bool = True
    min_capacity: int = 1
    max_capacity: int = 10
    target_cpu: int = 70
    target_memory: int = 80


@dataclass
class DesignSpecification:
    """Complete design specification for infrastructure."""
    project_name: str
    environment: str
    region: str
    services: List[ServiceRequirement]
    networking: NetworkingRequirement
    load_balancing: Optional[LoadBalancingRequirement] = None
    security: SecurityRequirement = None
    scaling: ScalingRequirement = None
    description: str = ""


def create_design_specification(
    project_name: str,
    environment: str,
    region: str,
    services: List[Dict[str, Any]],
    architecture_pattern: str,
    domain: Optional[str] = None,
    scaling_config: Optional[Dict[str, Any]] = None,
    description: str = ""
) -> str:
    """
    Create a structured design specification for infrastructure.
    
    Args:
        project_name: Name of the project
        environment: Environment (dev, staging, production)
        region: AWS region
        services: List of service requirements
        architecture_pattern: Architecture pattern to use
        domain: Domain name for certificates and DNS
        scaling_config: Auto-scaling configuration
        description: Description of the infrastructure
        
    Returns:
        JSON string representation of the design specification
    """
    
    # Convert services to ServiceRequirement objects
    service_requirements = []
    if services and isinstance(services, list):
        for service in services:
            # Ensure service is a dict and has required fields
            if not isinstance(service, dict):
                continue
            
            # Provide defaults for required fields
            service_name = service.get("name", "webapp")
            service_type = service.get("type", "ecs")
            
            try:
                req = ServiceRequirement(
                    name=service_name,
                    type=ServiceType(service_type),
                    description=service.get("description", f"{service_name} service"),
                    cpu=service.get("cpu"),
                    memory=service.get("memory"),
                    scaling=service.get("scaling"),
                    environment_variables=service.get("environment_variables"),
                    secrets=service.get("secrets"),
                    health_check_path=service.get("health_check_path", "/health"),
                    port=service.get("port"),
                    image=service.get("image"),
                    engine=service.get("engine"),
                    instance_class=service.get("instance_class"),
                    storage=service.get("storage")
                )
                service_requirements.append(req)
            except (ValueError, KeyError) as e:
                # Skip invalid service entries
                continue
    
    # If no valid services provided, create a default web service
    if not service_requirements:
        default_service = ServiceRequirement(
            name="webapp",
            type=ServiceType.ECS,
            description="React web application with Nginx",
            cpu=256,
            memory=512,
            health_check_path="/",
            port=80,
            image="nginx:latest"
        )
        service_requirements.append(default_service)
    
    # Create networking requirements
    networking = NetworkingRequirement(
        pattern=ArchitecturePattern(architecture_pattern),
        multi_az=True,
        public_subnets=True,
        private_subnets=True,
        database_subnets=any(s.type == ServiceType.RDS for s in service_requirements)
    )
    
    # Create load balancing requirements if needed
    load_balancing = None
    if any(s.type in [ServiceType.ECS, ServiceType.EC2] for s in service_requirements):
        routing_type = "path-based" if architecture_pattern == "microservices" else "simple"
        load_balancing = LoadBalancingRequirement(
            routing_type=routing_type,
            https_redirect=bool(domain)
        )
    
    # Create security requirements
    security = SecurityRequirement(
        https_required=bool(domain),
        certificate_domain=domain,
        additional_domains=[f"www.{domain}"] if domain else None
    )
    
    # Create scaling requirements
    scaling = ScalingRequirement(
        auto_scaling=True,
        min_capacity=scaling_config.get("min_capacity", 1) if scaling_config else 1,
        max_capacity=scaling_config.get("max_capacity", 10) if scaling_config else 10,
        target_cpu=scaling_config.get("target_cpu", 70) if scaling_config else 70,
        target_memory=scaling_config.get("target_memory", 80) if scaling_config else 80
    )
    
    # Create the complete specification
    spec = DesignSpecification(
        project_name=project_name,
        environment=environment,
        region=region,
        services=service_requirements,
        networking=networking,
        load_balancing=load_balancing,
        security=security,
        scaling=scaling,
        description=description
    )
    
    # Convert to JSON for handoff
    return json.dumps(asdict(spec), indent=2, default=str)


def validate_design_specification(spec_json: str) -> Dict[str, Any]:
    """
    Validate a design specification for completeness.
    
    Args:
        spec_json: JSON string of the design specification
        
    Returns:
        Validation result with issues and recommendations
    """
    
    try:
        spec_dict = json.loads(spec_json)
        issues = []
        recommendations = []
        
        # Check required fields
        required_fields = ["project_name", "environment", "region", "services"]
        for field in required_fields:
            if not spec_dict.get(field):
                issues.append(f"Missing required field: {field}")
        
        # Validate services
        services = spec_dict.get("services", [])
        if not services:
            issues.append("No services defined")
        
        for i, service in enumerate(services):
            if not service.get("name"):
                issues.append(f"Service {i} missing name")
            if not service.get("type"):
                issues.append(f"Service {i} missing type")
            
            # Type-specific validations
            if service.get("type") == "ecs":
                if not service.get("cpu"):
                    recommendations.append(f"Service {service.get('name', i)}: Consider specifying CPU allocation")
                if not service.get("memory"):
                    recommendations.append(f"Service {service.get('name', i)}: Consider specifying memory allocation")
                if not service.get("image"):
                    issues.append(f"ECS service {service.get('name', i)} missing container image")
            
            elif service.get("type") == "rds":
                if not service.get("engine"):
                    issues.append(f"RDS service {service.get('name', i)} missing engine type")
                if not service.get("instance_class"):
                    recommendations.append(f"RDS service {service.get('name', i)}: Consider specifying instance class")
        
        # Check networking
        networking = spec_dict.get("networking")
        if networking and networking.get("pattern") == "microservices":
            if len([s for s in services if s.get("type") == "ecs"]) < 2:
                recommendations.append("Microservices pattern with only one service - consider monolith pattern")
        
        # Check security
        security = spec_dict.get("security")
        if security and security.get("https_required") and not security.get("certificate_domain"):
            issues.append("HTTPS required but no certificate domain specified")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "recommendations": recommendations,
            "service_count": len(services),
            "architecture_pattern": networking.get("pattern") if networking else "unknown"
        }
        
    except json.JSONDecodeError:
        return {
            "valid": False,
            "issues": ["Invalid JSON format"],
            "recommendations": [],
            "service_count": 0,
            "architecture_pattern": "unknown"
        }


def suggest_architecture_improvements(spec_json: str) -> List[str]:
    """
    Suggest improvements to the design specification.
    
    Args:
        spec_json: JSON string of the design specification
        
    Returns:
        List of improvement suggestions
    """
    
    try:
        spec_dict = json.loads(spec_json)
        suggestions = []
        
        services = spec_dict.get("services", [])
        networking = spec_dict.get("networking", {})
        environment = spec_dict.get("environment", "")
        
        # Environment-specific suggestions
        if environment == "production":
            # Check for high availability
            if not networking.get("multi_az"):
                suggestions.append("Enable multi-AZ deployment for production high availability")
            
            # Check for auto-scaling
            scaling = spec_dict.get("scaling", {})
            if not scaling.get("auto_scaling"):
                suggestions.append("Enable auto-scaling for production workloads")
            
            # Check for appropriate instance sizes
            for service in services:
                if service.get("type") == "ecs":
                    cpu = service.get("cpu", 0)
                    if cpu < 512:
                        suggestions.append(f"Service {service.get('name')}: Consider larger CPU allocation for production")
                elif service.get("type") == "rds":
                    instance_class = service.get("instance_class", "")
                    if "micro" in instance_class:
                        suggestions.append(f"Database {service.get('name')}: Consider larger instance class for production")
        
        # Architecture pattern suggestions
        pattern = networking.get("pattern")
        service_count = len([s for s in services if s.get("type") in ["ecs", "ec2"]])
        
        if pattern == "microservices" and service_count == 1:
            suggestions.append("Single service with microservices pattern - consider monolith pattern for simplicity")
        elif pattern == "monolith" and service_count > 3:
            suggestions.append("Multiple services with monolith pattern - consider microservices pattern for better scalability")
        
        # Security suggestions
        security = spec_dict.get("security", {})
        if not security.get("https_required"):
            suggestions.append("Consider enabling HTTPS for better security")
        
        # Database suggestions
        rds_services = [s for s in services if s.get("type") == "rds"]
        if rds_services and not networking.get("database_subnets"):
            suggestions.append("Consider dedicated database subnets for better security")
        
        return suggestions
        
    except json.JSONDecodeError:
        return ["Unable to analyze specification due to invalid JSON format"]


# Tool functions that will be registered with the agent
def design_specification_tool(
    project_name: str = "",
    environment: str = "production",
    region: str = "us-east-1",
    services: str = "[]",
    architecture_pattern: str = "simple",
    domain: str = "",
    scaling_config: str = "{}",
    description: str = "",
    **kwargs
) -> str:
    """
    Create a structured design specification for infrastructure.
    Use this tool when you've analyzed user requirements and want to create
    a complete specification for handoff to the PaaS Manifest Generator.
    
    Args:
        project_name: Name of the project (default: "myapp")
        environment: Environment (dev, staging, production) (default: "production")
        region: AWS region (e.g., us-east-1) (default: "us-east-1")
        services: JSON string of service requirements (default: "[]")
        architecture_pattern: Architecture pattern (microservices, three-tier, monolith) (default: "simple")
        domain: Domain name for certificates and DNS (optional)
        scaling_config: JSON string of scaling configuration (optional, default: "{}")
        description: Description of the infrastructure
        **kwargs: Additional arguments (ignored for backward compatibility)
        
    Returns:
        Complete design specification in JSON format
    """
    
    try:
        # Set defaults for empty or missing values
        if not project_name:
            project_name = "myapp"
        if not environment:
            environment = "production"
        if not region:
            region = "us-east-1"
        if not services:
            services = "[]"
        if not architecture_pattern:
            architecture_pattern = "simple"
        if not scaling_config:
            scaling_config = "{}"
        
        services_list = json.loads(services)
        scaling_dict = json.loads(scaling_config) if scaling_config else {}
        
        spec = create_design_specification(
            project_name=project_name,
            environment=environment,
            region=region,
            services=services_list,
            architecture_pattern=architecture_pattern,
            domain=domain if domain else None,
            scaling_config=scaling_dict,
            description=description
        )
        
        # Validate the specification
        validation = validate_design_specification(spec)
        suggestions = suggest_architecture_improvements(spec)
        
        result = {
            "specification": json.loads(spec),
            "validation": validation,
            "suggestions": suggestions
        }
        
        return json.dumps(result, indent=2)
        
    except json.JSONDecodeError as e:
        return json.dumps({
            "error": f"Invalid JSON format in input: {str(e)}",
            "specification": None,
            "validation": {"valid": False, "issues": ["Invalid input format"]},
            "suggestions": []
        })
    except Exception as e:
        return json.dumps({
            "error": f"Failed to create specification: {str(e)}",
            "specification": None,
            "validation": {"valid": False, "issues": ["Specification creation failed"]},
            "suggestions": []
        }) 
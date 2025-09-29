"""PaaS generation tools for creating Cool Demo PaaS YAML configurations."""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import yaml

from paas_ai.utils.logging import get_logger

logger = get_logger("paas_ai.agents.tools.paas_generation_tools")


@dataclass
class YAMLFile:
    """Represents a generated YAML file."""

    filename: str
    content: str
    description: str


def generate_project_yaml(project_name: str, environment: str, region: str) -> str:
    """Generate project.yaml with basic metadata."""

    config = {"project": {"name": project_name, "environment": environment, "region": region}}

    return yaml.dump(config, default_flow_style=False, sort_keys=False)


def generate_networking_yaml(
    networking_spec: Dict[str, Any], security_groups: Optional[List[Dict[str, Any]]] = None
) -> str:
    """Generate networking.yaml with VPC, subnets, and security groups."""

    config = {}

    # VPC Configuration
    config["networking"] = {
        "vpc": {
            "cidr": networking_spec.get("vpc_cidr", "10.0.0.0/16"),
            "enable_dns_hostnames": True,
            "enable_dns_support": True,
        },
        "subnets": [],
    }

    # Generate subnets based on pattern
    pattern = networking_spec.get("pattern", "three-tier")
    multi_az = networking_spec.get("multi_az", True)

    if networking_spec.get("public_subnets", True):
        config["networking"]["subnets"].extend(
            [
                {
                    "name": "public-1",
                    "cidr": "10.0.1.0/24",
                    "availability_zone": "us-east-1a",
                    "public": True,
                }
            ]
        )
        if multi_az:
            config["networking"]["subnets"].append(
                {
                    "name": "public-2",
                    "cidr": "10.0.2.0/24",
                    "availability_zone": "us-east-1b",
                    "public": True,
                }
            )

    if networking_spec.get("private_subnets", True):
        if pattern == "microservices":
            base_cidr = "10.0.10.0/24"
            second_cidr = "10.0.11.0/24"
        else:
            base_cidr = "10.0.3.0/24"
            second_cidr = "10.0.4.0/24"

        config["networking"]["subnets"].extend(
            [
                {
                    "name": "private-1",
                    "cidr": base_cidr,
                    "availability_zone": "us-east-1a",
                    "public": False,
                }
            ]
        )
        if multi_az:
            config["networking"]["subnets"].append(
                {
                    "name": "private-2",
                    "cidr": second_cidr,
                    "availability_zone": "us-east-1b",
                    "public": False,
                }
            )

    if networking_spec.get("database_subnets", False):
        config["networking"]["subnets"].extend(
            [
                {
                    "name": "database-1",
                    "cidr": "10.0.20.0/24",
                    "availability_zone": "us-east-1a",
                    "public": False,
                }
            ]
        )
        if multi_az:
            config["networking"]["subnets"].append(
                {
                    "name": "database-2",
                    "cidr": "10.0.21.0/24",
                    "availability_zone": "us-east-1b",
                    "public": False,
                }
            )

    # Add security groups if provided
    if security_groups:
        config["security_groups"] = {}
        for sg in security_groups:
            config["security_groups"][sg["name"]] = {"rules": sg["rules"]}

    return yaml.dump(config, default_flow_style=False, sort_keys=False)


def generate_services_yaml(services: List[Dict[str, Any]], scaling_spec: Dict[str, Any]) -> str:
    """Generate services.yaml with ECS/EC2/RDS service definitions."""

    config = {"services": {}}

    for service in services:
        service_name = service["name"]
        service_type = service["type"]

        service_config = {"type": service_type}

        if service_type == "ecs":
            service_config.update(
                {
                    "cpu": service.get("cpu", 512),
                    "memory": service.get("memory", 1024),
                    "image": service.get("image", "nginx:latest"),
                    "port": service.get("port", 80),
                    "desired_count": service.get("desired_count", 2),
                }
            )

            # Add scaling configuration
            if scaling_spec.get("auto_scaling", True):
                service_config.update(
                    {
                        "min_capacity": scaling_spec.get("min_capacity", 1),
                        "max_capacity": scaling_spec.get("max_capacity", 10),
                        "scaling_policies": [
                            {
                                "metric": "cpu_utilization",
                                "target_value": scaling_spec.get("target_cpu", 70),
                                "scale_out_cooldown": 300,
                                "scale_in_cooldown": 300,
                            }
                        ],
                    }
                )

            # Add subnets for private deployment
            service_config["subnets"] = ["private-1", "private-2"]
            service_config["security_groups"] = [f"{service_name}-sg"]

            # Add health check
            health_check_path = service.get("health_check_path", "/health")
            service_config["health_check"] = {
                "command": [
                    "CMD",
                    "curl",
                    "-f",
                    f"http://localhost:{service.get('port', 80)}{health_check_path}",
                ],
                "interval": 30,
                "timeout": 5,
                "retries": 3,
            }

            # Add environment variables if specified
            if service.get("environment_variables"):
                service_config["environment"] = service["environment_variables"]

            # Add secrets if specified
            if service.get("secrets"):
                service_config["secrets"] = [
                    {"name": secret, "value": f"{secret.lower()}-secret"}
                    for secret in service["secrets"]
                ]

        elif service_type == "ec2":
            service_config.update(
                {
                    "instance_type": service.get("instance_type", "t3.small"),
                    "ami": service.get("ami", "amazon-linux-2"),
                    "key_pair": "production-key",
                    "security_groups": [f"{service_name}-sg"],
                }
            )

            # Add scaling for EC2
            if scaling_spec.get("auto_scaling", True):
                service_config.update(
                    {
                        "min_capacity": scaling_spec.get("min_capacity", 1),
                        "max_capacity": scaling_spec.get("max_capacity", 10),
                        "desired_count": 2,
                        "scaling_policies": [
                            {
                                "metric": "cpu_utilization",
                                "target_value": scaling_spec.get("target_cpu", 70),
                                "scale_out_cooldown": 300,
                                "scale_in_cooldown": 300,
                            }
                        ],
                    }
                )

            # Add user data for web servers
            if "web" in service_name.lower():
                service_config[
                    "user_data"
                ] = """#!/bin/bash
yum update -y
yum install -y nginx
systemctl start nginx
systemctl enable nginx"""

        elif service_type == "rds":
            service_config.update(
                {
                    "engine": service.get("engine", "postgres"),
                    "engine_version": service.get("engine_version", "13.7"),
                    "instance_class": service.get("instance_class", "db.t3.micro"),
                    "allocated_storage": service.get("storage", 20),
                    "multi_az": True,
                    "backup_retention_period": 7,
                    "backup_window": "03:00-04:00",
                    "maintenance_window": "sun:04:00-sun:05:00",
                    "subnets": ["database-1", "database-2"],
                    "security_groups": [f"{service_name}-sg"],
                }
            )

        config["services"][service_name] = service_config

    return yaml.dump(config, default_flow_style=False, sort_keys=False)


def generate_load_balancer_yaml(
    load_balancer_spec: Dict[str, Any], services: List[Dict[str, Any]]
) -> str:
    """Generate load-balancer.yaml with ALB configuration."""

    config = {"load_balancers": {}}

    alb_name = f"{load_balancer_spec.get('name', 'web')}-alb"
    routing_type = load_balancer_spec.get("routing_type", "simple")

    alb_config = {
        "type": "alb",
        "scheme": load_balancer_spec.get("scheme", "internet-facing"),
        "subnets": ["public-1", "public-2"],
        "security_groups": ["alb-sg"],
        "listeners": [],
    }

    # Add HTTP listener (redirect to HTTPS if enabled)
    if load_balancer_spec.get("https_redirect", True):
        alb_config["listeners"].append(
            {
                "port": 80,
                "protocol": "HTTP",
                "default_action": {
                    "type": "redirect",
                    "redirect_config": {
                        "protocol": "HTTPS",
                        "port": 443,
                        "status_code": "HTTP_301",
                    },
                },
            }
        )

        # Add HTTPS listener
        https_listener = {"port": 443, "protocol": "HTTPS", "certificate": "web-cert"}

        if routing_type == "path-based" and len(services) > 1:
            # Path-based routing for microservices
            https_listener["rules"] = []
            for service in services:
                if service["type"] in ["ecs", "ec2"]:
                    service_name = service["name"]
                    path = f"/{service_name.replace('-service', '')}/*"
                    https_listener["rules"].append(
                        {
                            "condition": {"path_pattern": path},
                            "action": {"type": "forward", "service": service_name},
                        }
                    )
        else:
            # Simple routing to first service
            first_service = next((s for s in services if s["type"] in ["ecs", "ec2"]), None)
            if first_service:
                https_listener["default_action"] = {
                    "type": "forward",
                    "service": first_service["name"],
                }

        alb_config["listeners"].append(https_listener)
    else:
        # HTTP only
        http_listener = {"port": 80, "protocol": "HTTP"}

        first_service = next((s for s in services if s["type"] in ["ecs", "ec2"]), None)
        if first_service:
            http_listener["default_action"] = {"type": "forward", "service": first_service["name"]}

        alb_config["listeners"].append(http_listener)

    config["load_balancers"][alb_name] = alb_config

    return yaml.dump(config, default_flow_style=False, sort_keys=False)


def generate_certificates_yaml(security_spec: Dict[str, Any]) -> str:
    """Generate certificates.yaml with SSL/TLS certificate configuration."""

    if not security_spec.get("certificate_domain"):
        return ""

    config = {"certificates": {}}

    cert_config = {
        "domain": security_spec["certificate_domain"],
        "validation_method": "DNS",
        "auto_validation": True,
        "auto_renewal": True,
    }

    # Add additional domains if specified
    additional_domains = security_spec.get("additional_domains", [])
    if additional_domains:
        cert_config["subject_alternative_names"] = additional_domains

    config["certificates"]["web-cert"] = cert_config

    return yaml.dump(config, default_flow_style=False, sort_keys=False)


def generate_dns_yaml(domain: str, load_balancer_name: str = "web-alb") -> str:
    """Generate dns.yaml with Route53 DNS records."""

    if not domain:
        return ""

    config = {
        "dns": {
            "zones": [
                {
                    "domain": domain,
                    "records": [
                        {
                            "name": "www",
                            "type": "A",
                            "alias": True,
                            "target": load_balancer_name,
                            "evaluate_target_health": True,
                        },
                        {
                            "name": "",
                            "type": "A",
                            "alias": True,
                            "target": load_balancer_name,
                            "evaluate_target_health": True,
                        },
                    ],
                }
            ]
        }
    }

    return yaml.dump(config, default_flow_style=False, sort_keys=False)


def generate_security_groups(services: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate security group configurations for services."""

    security_groups = []

    # ALB security group
    security_groups.append(
        {
            "name": "alb-sg",
            "rules": [
                {
                    "type": "ingress",
                    "protocol": "tcp",
                    "port": 80,
                    "source": "0.0.0.0/0",
                    "description": "HTTP from anywhere",
                },
                {
                    "type": "ingress",
                    "protocol": "tcp",
                    "port": 443,
                    "source": "0.0.0.0/0",
                    "description": "HTTPS from anywhere",
                },
                {
                    "type": "egress",
                    "protocol": "all",
                    "port": "all",
                    "destination": "0.0.0.0/0",
                    "description": "All outbound traffic",
                },
            ],
        }
    )

    # Service-specific security groups
    for service in services:
        service_name = service["name"]
        service_type = service["type"]

        if service_type in ["ecs", "ec2"]:
            port = service.get("port", 80)
            security_groups.append(
                {
                    "name": f"{service_name}-sg",
                    "rules": [
                        {
                            "type": "ingress",
                            "protocol": "tcp",
                            "port": port,
                            "source": "alb-sg",
                            "description": f"HTTP from ALB to {service_name}",
                        },
                        {
                            "type": "egress",
                            "protocol": "all",
                            "port": "all",
                            "destination": "0.0.0.0/0",
                            "description": "All outbound traffic",
                        },
                    ],
                }
            )

        elif service_type == "rds":
            port = 5432 if service.get("engine", "postgres") == "postgres" else 3306
            security_groups.append(
                {
                    "name": f"{service_name}-sg",
                    "rules": [
                        {
                            "type": "ingress",
                            "protocol": "tcp",
                            "port": port,
                            "source": "app-sg",
                            "description": f"Database access from application servers",
                        }
                    ],
                }
            )

    return security_groups


def generate_paas_manifests(design_spec_json: str) -> str:
    """
    Generate complete Cool Demo PaaS YAML manifests from design specification.

    Args:
        design_spec_json: JSON string of the design specification

    Returns:
        JSON string containing all generated YAML files
    """

    try:
        spec = json.loads(design_spec_json)
        files = []

        # Debug: Log the input structure for troubleshooting
        logger.debug(
            f"Received specification structure: {list(spec.keys()) if isinstance(spec, dict) else type(spec)}"
        )

        # Handle both direct spec and wrapped spec formats
        if "specification" in spec:
            logger.debug("Found wrapped specification format")
            spec = spec["specification"]
        else:
            logger.debug("Using direct specification format")

        # Debug: Log the final spec structure
        logger.debug(
            f"Final specification keys: {list(spec.keys()) if isinstance(spec, dict) else type(spec)}"
        )

        # Extract specification components with enhanced error messages
        try:
            project_name = spec["project_name"]
            logger.debug(f"Successfully extracted project_name: {project_name}")
        except KeyError as e:
            logger.error(
                f"Missing required field 'project_name'. Available keys: {list(spec.keys())}"
            )
            raise KeyError(
                f"Missing required field 'project_name'. Available keys: {list(spec.keys())}"
            ) from e

        try:
            environment = spec["environment"]
            region = spec["region"]
            services = spec["services"]
            networking = spec["networking"]
        except KeyError as e:
            logger.error(
                f"Missing required field '{e.args[0]}'. Available keys: {list(spec.keys())}"
            )
            raise KeyError(
                f"Missing required field '{e.args[0]}'. Available keys: {list(spec.keys())}"
            ) from e

        load_balancing = spec.get("load_balancing")
        security = spec.get("security", {})
        scaling = spec.get("scaling", {})

        # Generate project.yaml
        project_content = generate_project_yaml(project_name, environment, region)
        files.append(
            YAMLFile(
                filename="project.yaml",
                content=project_content,
                description="Project metadata and basic configuration",
            )
        )

        # Generate security groups
        security_groups = generate_security_groups(services)

        # Generate networking.yaml
        networking_content = generate_networking_yaml(networking, security_groups)
        files.append(
            YAMLFile(
                filename="networking.yaml",
                content=networking_content,
                description="VPC, subnets, and security groups configuration",
            )
        )

        # Generate services.yaml
        services_content = generate_services_yaml(services, scaling)
        files.append(
            YAMLFile(
                filename="services.yaml",
                content=services_content,
                description="ECS, EC2, and RDS service definitions",
            )
        )

        # Generate load-balancer.yaml if needed
        if load_balancing and any(s["type"] in ["ecs", "ec2"] for s in services):
            lb_content = generate_load_balancer_yaml(load_balancing, services)
            files.append(
                YAMLFile(
                    filename="load-balancer.yaml",
                    content=lb_content,
                    description="Application Load Balancer configuration",
                )
            )

        # Generate certificates.yaml if HTTPS is required
        if security.get("https_required") and security.get("certificate_domain"):
            cert_content = generate_certificates_yaml(security)
            if cert_content:
                files.append(
                    YAMLFile(
                        filename="certificates.yaml",
                        content=cert_content,
                        description="SSL/TLS certificate management",
                    )
                )

        # Generate dns.yaml if domain is specified
        domain = security.get("certificate_domain")
        if domain:
            dns_content = generate_dns_yaml(domain)
            if dns_content:
                files.append(
                    YAMLFile(
                        filename="dns.yaml", content=dns_content, description="Route53 DNS records"
                    )
                )

        # Prepare result
        result = {
            "files": [
                {"filename": f.filename, "content": f.content, "description": f.description}
                for f in files
            ],
            "summary": {
                "total_files": len(files),
                "services_count": len(services),
                "architecture_pattern": networking.get("pattern", "unknown"),
                "has_load_balancer": bool(load_balancing),
                "has_https": security.get("https_required", False),
                "has_database": any(s["type"] == "rds" for s in services),
            },
            "deployment_notes": [
                f"Generated {len(files)} configuration files for {project_name}",
                f"Architecture pattern: {networking.get('pattern', 'unknown')}",
                f"Environment: {environment}",
                f"Region: {region}",
                "All files are ready for deployment with Cool Demo PaaS",
            ],
        }

        return json.dumps(result, indent=2)

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in generate_paas_manifests: {e}")
        logger.error(f"Input data (first 500 chars): {design_spec_json[:500]}")
        return json.dumps(
            {
                "error": f"Invalid JSON in design specification: {str(e)}",
                "files": [],
                "summary": {},
                "deployment_notes": [],
                "debug_info": {
                    "error_type": "json_decode_error",
                    "input_preview": design_spec_json[:200] + "..."
                    if len(design_spec_json) > 200
                    else design_spec_json,
                    "error_position": getattr(e, "pos", "unknown"),
                },
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error in generate_paas_manifests: {e}")
        logger.error(f"Input data type: {type(design_spec_json)}")
        logger.error(f"Input data (first 500 chars): {str(design_spec_json)[:500]}")
        return json.dumps(
            {
                "error": f"Failed to generate manifests: {str(e)}",
                "files": [],
                "summary": {},
                "deployment_notes": [],
                "debug_info": {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "input_type": str(type(design_spec_json)),
                    "input_preview": str(design_spec_json)[:200] + "..."
                    if len(str(design_spec_json)) > 200
                    else str(design_spec_json),
                },
            }
        )


def validate_generated_manifests(manifests_json: str) -> str:
    """
    Validate generated PaaS manifests for completeness and correctness.

    Args:
        manifests_json: JSON string of generated manifests

    Returns:
        JSON string with validation results
    """

    try:
        manifests = json.loads(manifests_json)
        files = manifests.get("files", [])
        issues = []
        recommendations = []

        # Check required files
        filenames = [f["filename"] for f in files]
        required_files = ["project.yaml", "networking.yaml", "services.yaml"]

        for required in required_files:
            if required not in filenames:
                issues.append(f"Missing required file: {required}")

        # Validate YAML syntax
        for file_info in files:
            try:
                yaml.safe_load(file_info["content"])
            except yaml.YAMLError as e:
                issues.append(f"Invalid YAML syntax in {file_info['filename']}: {str(e)}")

        # Check for logical consistency
        has_services = "services.yaml" in filenames
        has_load_balancer = "load-balancer.yaml" in filenames
        has_certificates = "certificates.yaml" in filenames
        has_dns = "dns.yaml" in filenames

        if has_services and not has_load_balancer:
            recommendations.append("Consider adding load balancer for better availability")

        if has_load_balancer and not has_certificates:
            recommendations.append("Consider adding HTTPS certificates for security")

        if has_certificates and not has_dns:
            recommendations.append("Consider adding DNS records for domain management")

        # Validate file contents
        for file_info in files:
            if file_info["filename"] == "services.yaml":
                try:
                    content = yaml.safe_load(file_info["content"])
                    services = content.get("services", {})
                    if not services:
                        issues.append("services.yaml contains no services")

                    for service_name, service_config in services.items():
                        if not service_config.get("type"):
                            issues.append(f"Service {service_name} missing type")
                except:
                    pass  # YAML parsing error already caught above

        return json.dumps(
            {
                "valid": len(issues) == 0,
                "issues": issues,
                "recommendations": recommendations,
                "file_count": len(files),
                "completeness_score": max(0, 100 - len(issues) * 20 - len(recommendations) * 5),
            },
            indent=2,
        )

    except json.JSONDecodeError:
        return json.dumps(
            {
                "valid": False,
                "issues": ["Invalid JSON format"],
                "recommendations": [],
                "file_count": 0,
                "completeness_score": 0,
            }
        )


# Tool functions for the PaaS Manifest Generator agent
def paas_manifest_generator_tool(design_specification: str, **kwargs) -> str:
    """
    Generate complete Cool Demo PaaS YAML manifests from a structured JSON design specification.

    IMPORTANT: This tool expects a JSON specification, NOT natural language.

    Your workflow should be:
    1. Receive natural language design from Designer Agent
    2. Use RAG to understand platform capabilities for each service mentioned
    3. Intelligently map the natural language requirements to structured JSON format
    4. Call this tool with the structured JSON specification

    The JSON should have these fields:
    - project_name, environment, region
    - services: array of service objects with type, requirements, etc.
    - networking, load_balancing, security, scaling configurations

    Args:
        design_specification: Structured JSON specification (NOT natural language)
        **kwargs: Additional arguments (ignored for backward compatibility)

    Returns:
        JSON string containing all generated YAML files with descriptions
    """
    from langgraph.types import interrupt

    if not design_specification or not isinstance(design_specification, str):
        # Interrupt immediately for human assistance on invalid input
        human_response = interrupt(
            {
                "message": f"Invalid design specification provided: {design_specification}. Need help with the correct format."
            }
        )
        # Process human response and return corrected result
        if isinstance(human_response, dict) and "data" in human_response:
            # Human provided guidance, try again with their input
            return paas_manifest_generator_tool(human_response["data"])

    try:
        # For now, the underlying generate_paas_manifests still expects JSON
        # The agent should have used RAG to convert natural language to structured JSON
        # before calling this tool. If they pass natural language directly, that's an error in their process.

        result = generate_paas_manifests(design_specification)

        # Check if generation failed
        manifests = json.loads(result)
        if "error" in manifests:
            # Interrupt immediately for human assistance on any error
            human_response = interrupt(
                {
                    "message": f"PaaS manifest generation failed with error: {manifests['error']}. Input was: {design_specification[:200]}... Can you help fix this?"
                }
            )

            # Process human response
            if isinstance(human_response, dict) and "data" in human_response:
                return paas_manifest_generator_tool(human_response["data"])
            else:
                return json.dumps(
                    {
                        "error": f"Generation failed: {manifests['error']}",
                        "files": [],
                        "summary": {},
                        "deployment_notes": [],
                    }
                )

        # Success - validate and return
        validation = validate_generated_manifests(result)
        manifests["validation"] = validation
        return json.dumps(manifests)

    except Exception as e:
        logger.error(f"Error in paas_manifest_generator_tool: {e}")

        # Interrupt immediately for human assistance on any exception
        human_response = interrupt(
            {
                "message": f"PaaS manifest generation failed with technical error: {str(e)}. Input was: {design_specification[:200]}... Can you help resolve this?"
            }
        )

        # Process human response
        if isinstance(human_response, dict) and "data" in human_response:
            return paas_manifest_generator_tool(human_response["data"])
        else:
            return json.dumps(
                {
                    "error": f"Failed to generate manifests: {str(e)}",
                    "files": [],
                    "summary": {},
                    "deployment_notes": [],
                }
            )


def manifest_validation_tool(manifest_data: str, **kwargs) -> str:
    """
    Validate generated PaaS manifests for completeness and correctness.
    Use this tool to check your generated configurations before final delivery.

    Args:
        manifest_data: JSON string of generated manifests to validate
        **kwargs: Additional arguments (ignored for backward compatibility)

    Returns:
        JSON string with validation results and recommendations
    """
    if not manifest_data or not isinstance(manifest_data, str):
        return json.dumps(
            {
                "valid": False,
                "issues": ["Invalid manifest data provided. Must be a non-empty JSON string."],
                "recommendations": [],
                "file_count": 0,
                "completeness_score": 0,
            }
        )

    try:
        return validate_generated_manifests(manifest_data)
    except Exception as e:
        return json.dumps(
            {
                "valid": False,
                "issues": [f"Validation failed: {str(e)}"],
                "recommendations": [],
                "file_count": 0,
                "completeness_score": 0,
            }
        )

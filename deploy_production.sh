#!/bin/bash

# Production deployment script for FinanceRAG-Pro
# This script sets up the production environment with all necessary configurations

set -e  # Exit on any error

echo "=============================================="
echo "FinanceRAG-Pro Production Deployment Script"
echo "=============================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if Docker and Docker Compose are installed
check_dependencies() {
    print_header "Checking dependencies..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_status "Docker and Docker Compose are installed."
}

# Create necessary directories
create_directories() {
    print_header "Creating necessary directories..."
    
    directories=(
        "data"
        "logs" 
        "nginx/conf.d"
        "nginx/ssl"
        "db/init"
        "monitoring"
        "monitoring/grafana/dashboards"
        "monitoring/grafana/datasources"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        print_status "Created directory: $dir"
    done
}

# Create environment file
create_env_file() {
    print_header "Creating environment configuration..."
    
    if [ ! -f .env.production ]; then
        cat > .env.production << EOF
# Production Environment Configuration
ENVIRONMENT=production

# Database Configuration
POSTGRES_PASSWORD=your_secure_postgres_password_here
PG_HOST=postgres
PG_PORT=5432
PG_DATABASE=financerag
PG_USER=postgres

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379

# MinIO Configuration
MINIO_ACCESS_KEY=your_minio_access_key_here
MINIO_SECRET_KEY=your_minio_secret_key_here
MINIO_ENDPOINT=minio:9000

# API Keys
OPENAI_API_KEY=your_openai_api_key_here

# Monitoring
GRAFANA_PASSWORD=your_grafana_password_here

# Security
JWT_SECRET=your_jwt_secret_here
ENCRYPTION_KEY=your_encryption_key_here

# Application Settings
LOG_LEVEL=INFO
DEBUG=false
CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com

# SSL Configuration (if using HTTPS)
SSL_CERT_PATH=/etc/nginx/ssl/cert.pem
SSL_KEY_PATH=/etc/nginx/ssl/key.pem
EOF
        print_warning "Created .env.production file. Please update it with your actual values!"
    else
        print_status ".env.production file already exists."
    fi
}

# Create nginx configuration
create_nginx_config() {
    print_header "Creating Nginx configuration..."
    
    cat > nginx/conf.d/default.conf << 'EOF'
upstream financerag_backend {
    server financerag-pro:8000;
}

# HTTP server (redirects to HTTPS in production)
server {
    listen 80;
    server_name localhost;
    
    # Health check endpoint
    location /health {
        proxy_pass http://financerag_backend/health;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Main application
    location / {
        proxy_pass http://financerag_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeout settings
        proxy_connect_timeout       60s;
        proxy_send_timeout          60s;
        proxy_read_timeout          60s;
        
        # Buffer settings
        proxy_buffer_size           4k;
        proxy_buffers               4 32k;
        proxy_busy_buffers_size     64k;
        
        # Large file upload support
        client_max_body_size        50M;
        client_body_buffer_size     128k;
    }
    
    # Static files (if any)
    location /static/ {
        alias /app/static/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
}
EOF
    
    print_status "Created Nginx configuration."
}

# Create Prometheus configuration
create_monitoring_config() {
    print_header "Creating monitoring configuration..."
    
    cat > monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'financerag-app'
    static_configs:
      - targets: ['financerag-pro:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
    scrape_interval: 30s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
    scrape_interval: 30s
EOF
    
    print_status "Created Prometheus configuration."
}

# Build and deploy
deploy() {
    print_header "Building and deploying application..."
    
    # Load environment variables
    if [ -f .env.production ]; then
        export $(cat .env.production | grep -v '^#' | xargs)
    fi
    
    # Build the production image
    print_status "Building production Docker image..."
    docker-compose -f docker-compose.production.yml build
    
    # Start the services
    print_status "Starting services..."
    docker-compose -f docker-compose.production.yml up -d
    
    # Wait for services to be healthy
    print_status "Waiting for services to be healthy..."
    sleep 30
    
    # Check service health
    check_health
}

# Check service health
check_health() {
    print_header "Checking service health..."
    
    services=("financerag-pro" "postgres" "redis" "minio")
    
    for service in "${services[@]}"; do
        if docker-compose -f docker-compose.production.yml ps $service | grep -q "healthy\|Up"; then
            print_status "$service is running"
        else
            print_warning "$service might have issues"
        fi
    done
    
    # Test application endpoint
    if curl -f -s http://localhost/health > /dev/null; then
        print_status "Application health check passed"
    else
        print_warning "Application health check failed"
    fi
}

# Show deployment status
show_status() {
    print_header "Deployment Status"
    
    echo ""
    echo "Services:"
    docker-compose -f docker-compose.production.yml ps
    
    echo ""
    echo "Application URLs:"
    echo "- Main Application: http://localhost"
    echo "- API Documentation: http://localhost/docs"
    echo "- Health Check: http://localhost/health"
    echo "- MinIO Console: http://localhost:9001"
    echo "- Grafana (if monitoring enabled): http://localhost:3000"
    echo "- Prometheus (if monitoring enabled): http://localhost:9090"
    
    echo ""
    print_status "Deployment completed successfully!"
    print_warning "Remember to:"
    echo "  1. Update .env.production with your actual values"
    echo "  2. Configure SSL certificates for production"
    echo "  3. Set up proper domain and DNS"
    echo "  4. Configure firewall rules"
    echo "  5. Set up backup procedures"
}

# Cleanup function
cleanup() {
    print_header "Cleaning up..."
    docker-compose -f docker-compose.production.yml down
    docker system prune -f
    print_status "Cleanup completed."
}

# Main deployment flow
main() {
    case "${1:-deploy}" in
        "deploy")
            check_dependencies
            create_directories
            create_env_file
            create_nginx_config
            create_monitoring_config
            deploy
            show_status
            ;;
        "status")
            show_status
            ;;
        "health")
            check_health
            ;;
        "cleanup")
            cleanup
            ;;
        "restart")
            print_header "Restarting services..."
            docker-compose -f docker-compose.production.yml restart
            check_health
            ;;
        *)
            echo "Usage: $0 {deploy|status|health|cleanup|restart}"
            echo ""
            echo "Commands:"
            echo "  deploy   - Full deployment (default)"
            echo "  status   - Show deployment status"
            echo "  health   - Check service health"
            echo "  cleanup  - Stop and clean up"
            echo "  restart  - Restart services"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
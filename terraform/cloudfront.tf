locals {
  cf_origin_id     = "EC2Origin"
  ec2_origin_dns   = aws_instance.credit_guard.public_dns
  cert_arn         = aws_acm_certificate_validation.wildcard.certificate_arn
  waf_arn          = aws_wafv2_web_acl.credit_guard.arn
  all_methods      = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
  readonly_methods = ["GET", "HEAD", "OPTIONS"]
  cached_methods   = ["GET", "HEAD"]
}

# CloudFront Function que bloqueia escrita/deleção externa na UI do MLflow
# permitindo apenas leitura (GET/HEAD/OPTIONS) e buscas da UI (/search, /get)
resource "aws_cloudfront_function" "mlflow_readonly" {
  name    = "${var.project_name}-mlflow-readonly"
  runtime = "cloudfront-js-2.0"
  comment = "Read-only guard for public MLflow UI"
  publish = true
  code    = <<-EOT
    function handler(event) {
      var req = event.request;
      var m = req.method;
      var uri = req.uri;
      if (m === 'GET' || m === 'HEAD' || m === 'OPTIONS') {
        return req;
      }
      if (m === 'POST' && (uri.indexOf('/search') !== -1 || uri.indexOf('/get') !== -1 || uri.indexOf('graphql') !== -1)) {
        return req;
      }
      return {
        statusCode: 403,
        statusDescription: 'Forbidden',
        headers: { 'content-type': { value: 'application/json' } },
        body: '{"error":"Read-only public mode enabled for MLflow UI"}'
      };
    }
  EOT
}

resource "aws_cloudfront_distribution" "api" {
  enabled     = true
  aliases     = ["api.${var.base_domain}"]
  web_acl_id  = local.waf_arn
  price_class = "PriceClass_100"
  comment     = "CreditGuard API — Fase 4"

  origin {
    domain_name = local.ec2_origin_dns
    origin_id   = local.cf_origin_id

    custom_origin_config {
      http_port              = 8000
      https_port             = 443
      origin_protocol_policy = "http-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }

  default_cache_behavior {
    target_origin_id       = local.cf_origin_id
    viewer_protocol_policy = "redirect-to-https"
    allowed_methods        = local.all_methods
    cached_methods         = local.cached_methods
    compress               = true

    forwarded_values {
      query_string = true
      headers      = ["Authorization", "Content-Type", "Accept"]
      cookies { forward = "none" }
    }

    min_ttl     = 0
    default_ttl = 0
    max_ttl     = 0
  }

  restrictions {
    geo_restriction {
      restriction_type = "whitelist"
      locations        = var.allowed_countries
    }
  }

  viewer_certificate {
    acm_certificate_arn      = local.cert_arn
    ssl_support_method       = "sni-only"
    minimum_protocol_version = "TLSv1.2_2021"
  }

  tags = {
    Name    = "${var.project_name}-cf-api"
    Project = var.project_name
  }
}

resource "aws_cloudfront_distribution" "mlflow" {
  enabled     = true
  aliases     = ["mlflow.${var.base_domain}"]
  web_acl_id  = local.waf_arn
  price_class = "PriceClass_100"
  comment     = "MLflow Tracking UI — Fase 4 (Read-Only Public)"

  origin {
    domain_name = local.ec2_origin_dns
    origin_id   = local.cf_origin_id

    custom_origin_config {
      http_port              = 5000
      https_port             = 443
      origin_protocol_policy = "http-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }

  default_cache_behavior {
    target_origin_id       = local.cf_origin_id
    viewer_protocol_policy = "redirect-to-https"
    allowed_methods        = local.all_methods
    cached_methods         = local.cached_methods
    compress               = true

    function_association {
      event_type   = "viewer-request"
      function_arn = aws_cloudfront_function.mlflow_readonly.arn
    }

    forwarded_values {
      query_string = true
      headers      = ["*"]
      cookies { forward = "all" }
    }

    min_ttl     = 0
    default_ttl = 0
    max_ttl     = 30
  }

  restrictions {
    geo_restriction {
      restriction_type = "whitelist"
      locations        = var.allowed_countries
    }
  }

  viewer_certificate {
    acm_certificate_arn      = local.cert_arn
    ssl_support_method       = "sni-only"
    minimum_protocol_version = "TLSv1.2_2021"
  }

  tags = {
    Name    = "${var.project_name}-cf-mlflow"
    Project = var.project_name
  }
}

resource "aws_cloudfront_distribution" "grafana" {
  enabled     = true
  aliases     = ["grafana.${var.base_domain}"]
  web_acl_id  = local.waf_arn
  price_class = "PriceClass_100"
  comment     = "Grafana — Fase 4 (Read-Only Anonymous Viewer)"

  origin {
    domain_name = local.ec2_origin_dns
    origin_id   = local.cf_origin_id

    custom_origin_config {
      http_port              = 3000
      https_port             = 443
      origin_protocol_policy = "http-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }

  default_cache_behavior {
    target_origin_id       = local.cf_origin_id
    viewer_protocol_policy = "redirect-to-https"
    allowed_methods        = local.all_methods
    cached_methods         = local.cached_methods
    compress               = true

    forwarded_values {
      query_string = true
      headers      = ["*"]
      cookies { forward = "all" }
    }

    min_ttl     = 0
    default_ttl = 0
    max_ttl     = 60
  }

  restrictions {
    geo_restriction {
      restriction_type = "whitelist"
      locations        = var.allowed_countries
    }
  }

  viewer_certificate {
    acm_certificate_arn      = local.cert_arn
    ssl_support_method       = "sni-only"
    minimum_protocol_version = "TLSv1.2_2021"
  }

  tags = {
    Name    = "${var.project_name}-cf-grafana"
    Project = var.project_name
  }
}

resource "aws_cloudfront_distribution" "prometheus" {
  enabled     = true
  aliases     = ["prometheus.${var.base_domain}"]
  web_acl_id  = local.waf_arn
  price_class = "PriceClass_100"
  comment     = "Prometheus — Fase 4 (Read-Only)"

  origin {
    domain_name = local.ec2_origin_dns
    origin_id   = local.cf_origin_id

    custom_origin_config {
      http_port              = 9090
      https_port             = 443
      origin_protocol_policy = "http-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }

  default_cache_behavior {
    target_origin_id       = local.cf_origin_id
    viewer_protocol_policy = "redirect-to-https"
    allowed_methods        = local.readonly_methods
    cached_methods         = local.cached_methods
    compress               = true

    forwarded_values {
      query_string = true
      headers      = ["*"]
      cookies { forward = "none" }
    }

    min_ttl     = 0
    default_ttl = 0
    max_ttl     = 30
  }

  restrictions {
    geo_restriction {
      restriction_type = "whitelist"
      locations        = var.allowed_countries
    }
  }

  viewer_certificate {
    acm_certificate_arn      = local.cert_arn
    ssl_support_method       = "sni-only"
    minimum_protocol_version = "TLSv1.2_2021"
  }

  tags = {
    Name    = "${var.project_name}-cf-prometheus"
    Project = var.project_name
  }
}

resource "aws_cloudfront_distribution" "evidently" {
  enabled     = true
  aliases     = ["evidently.${var.base_domain}"]
  web_acl_id  = local.waf_arn
  price_class = "PriceClass_100"
  comment     = "Evidently AI Official UI — Fase 4 (Read-Only)"

  origin {
    domain_name = local.ec2_origin_dns
    origin_id   = local.cf_origin_id

    custom_origin_config {
      http_port              = 8085
      https_port             = 443
      origin_protocol_policy = "http-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }

  default_cache_behavior {
    target_origin_id       = local.cf_origin_id
    viewer_protocol_policy = "redirect-to-https"
    allowed_methods        = local.readonly_methods
    cached_methods         = local.cached_methods
    compress               = true

    forwarded_values {
      query_string = true
      headers      = ["*"]
      cookies { forward = "none" }
    }

    min_ttl     = 0
    default_ttl = 0
    max_ttl     = 0
  }

  restrictions {
    geo_restriction {
      restriction_type = "whitelist"
      locations        = var.allowed_countries
    }
  }

  viewer_certificate {
    acm_certificate_arn      = local.cert_arn
    ssl_support_method       = "sni-only"
    minimum_protocol_version = "TLSv1.2_2021"
  }

  tags = {
    Name    = "${var.project_name}-cf-evidently"
    Project = var.project_name
  }
}

# AI Palette Generator API

A sophisticated color palette generation API using AI models and traditional color theory methods.

## Features

- **AI-Powered Generation**: Using Gaussian Mixture Models and Deep Learning
- **Traditional Methods**: Complementary, Triadic, Analogous, Monochromatic palettes
- **Style Variations**: Balanced, Vibrant, Muted, Pastel styles
- **Accessibility Checking**: WCAG contrast ratio validation
- **Multiple Export Formats**: JSON API responses

## API Endpoints

- `GET /` - API information
- `POST /generate-palette` - Generate color palettes
- `POST /check-contrast` - Check WCAG compliance
- `GET /health` - Health check
- `GET /methods` - Available generation methods

## Deployment

Deployed on Railway.app with automatic scaling and monitoring.

## Usage

```bash
curl -X POST "YOUR_RAILWAY_URL/generate-palette" \
     -H "Content-Type: application/json" \
     -d '{
       "method": "gmm",
       "num_colors": 5,
       "style": "vibrant"
     }'
# palette-generator-api

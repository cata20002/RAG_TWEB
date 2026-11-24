import {
  AngularNodeAppEngine,
  createNodeRequestHandler,
  isMainModule,
  writeResponseToNodeResponse,
} from '@angular/ssr/node';
import express from 'express';
import { join } from 'node:path';

const browserDistFolder = join(import.meta.dirname, '../browser');

const app = express();
const angularApp = new AngularNodeAppEngine();

/**
 * Parse JSON request bodies for API endpoints.
 */
app.use(express.json());

/**
 * Feedback API endpoint.
 * Accepts feedback on AI responses (rating, optional comment, question, answer).
 */
app.post('/api/feedback', async (req, res) => {
  try {
    const { rating, comment, question, answer } = req.body;

    if (!rating) {
      res.status(400).json({ error: 'rating is required' });
      return;
    }

    // Forward feedback to the Python backend
    const backendUrl = process.env['BACKEND_URL'] || 'http://localhost:8000';
    const response = await fetch(`${backendUrl}/api/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ rating, comment, question, answer }),
    });

    if (!response.ok) {
      const errorBody = await response.text();
      res.status(response.status).json({ error: errorBody });
      return;
    }

    const data = await response.json();
    res.json(data);
  } catch (error) {
    console.error('Feedback endpoint error:', error);
    res.status(500).json({ error: 'Failed to submit feedback' });
  }
});

/**
 * Serve static files from /browser
 */
app.use(
  express.static(browserDistFolder, {
    maxAge: '1y',
    index: false,
    redirect: false,
  }),
);

/**
 * Handle all other requests by rendering the Angular application.
 */
app.use((req, res, next) => {
  angularApp
    .handle(req)
    .then((response) =>
      response ? writeResponseToNodeResponse(response, res) : next(),
    )
    .catch(next);
});

/**
 * Start the server if this module is the main entry point, or it is ran via PM2.
 * The server listens on the port defined by the `PORT` environment variable, or defaults to 4000.
 */
if (isMainModule(import.meta.url) || process.env['pm_id']) {
  const port = process.env['PORT'] || 4000;
  app.listen(port, (error) => {
    if (error) {
      throw error;
    }

    console.log(`Node Express server listening on http://localhost:${port}`);
  });
}

/**
 * Request handler used by the Angular CLI (for dev-server and during build) or Firebase Cloud Functions.
 */
export const reqHandler = createNodeRequestHandler(app);

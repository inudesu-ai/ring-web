import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import helmet from 'helmet';

const app = express();
const port = Number(process.env.PORT || 3000);

app.use(helmet());
app.use(cors());
app.use(express.json({ limit: '1mb' }));

app.get('/health', (_req, res) => {
  res.json({ ok: true, service: 'ring-api', time: new Date().toISOString() });
});

app.get('/hello', (_req, res) => {
  res.json({ message: 'hello from api.inudesu.xyz' });
});

app.post('/v1/echo', (req, res) => {
  res.json({ received: req.body ?? null });
});

app.listen(port, '127.0.0.1', () => {
  console.log(`ring-api listening on http://127.0.0.1:${port}`);
});

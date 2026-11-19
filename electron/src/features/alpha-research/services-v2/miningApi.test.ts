import { beforeEach, describe, expect, it, vi } from 'vitest';

const client = vi.hoisted(() => ({ get: vi.fn(), post: vi.fn() }));
vi.mock('../../../services/aiStrategyClients', () => ({ apiClient: client }));

import { previewMining, startMining } from './api';

describe('factor mining API', () => {
  beforeEach(() => vi.clearAllMocks());

  it('defaults new mining requests to all A-shares', async () => {
    client.post.mockResolvedValue({ data: { code: 200, data: { algorithm: { key: 'rdagent' } } } });
    await previewMining({ direction: '价量假设', algorithm: 'rdagent' });
    expect(client.post.mock.calls[0][1]).toMatchObject({ universe: 'all_a' });
  });

  it('submits statistical mining as an idempotent JSON request with frozen scope', async () => {
    client.post.mockResolvedValue({
      data: {
        code: 200,
        data: { task_id: 'stat-task-1', status: 'queued', queue_position: 2 },
      },
    });

    const result = await startMining({
      direction: '',
      algorithm: 'statistical',
      miningStart: '2020-01-01',
      miningEnd: '2022-12-31',
      validStart: '2023-01-01',
      validEnd: '2023-12-31',
      market: 'a_share',
      universe: 'csi500',
      dataSource: 'parquet',
      freq: 'day',
    }, 'stat-request-1');

    expect(client.post).toHaveBeenCalledTimes(1);
    const [url, body, config] = client.post.mock.calls[0];
    expect(url).toBe('/alpha-agent/evolve');
    expect(body).toEqual({
      algorithm: 'statistical',
      market: 'a_share',
      universe: 'csi500',
      data_source: 'qlib_bin',
      freq: 'day',
      mining_start: '2020-01-01',
      mining_end: '2022-12-31',
      valid_start: '2023-01-01',
      valid_end: '2023-12-31',
      auto_materialize: true,
    });
    expect(config.headers['Idempotency-Key']).toEqual(expect.any(String));
    expect(config.headers['Idempotency-Key'].length).toBeGreaterThan(0);
    expect(result.data?.taskId).toBe('stat-task-1');
  });

  it('submits the RD-Agent dataset selection in its durable request body', async () => {
    client.post.mockResolvedValue({
      data: { code: 200, data: { task_id: 'rd-task-1', status: 'queued' } },
    });
    await startMining({
      direction: '价量假设',
      algorithm: 'rdagent',
      market: 'a_share',
      universe: 'csi300',
      datasets: ['l1_factors'],
      freq: 'day',
      miningStart: '2020-01-01',
      miningEnd: '2022-12-31',
      validStart: '2023-01-01',
      validEnd: '2023-12-31',
    }, 'rd-request-1');
    const [url, body, config] = client.post.mock.calls[0];
    expect(url).toBe('/alpha-agent/evolve');
    expect(body).toMatchObject({
      algorithm: 'rdagent',
      freq: 'day',
      datasets: ['l1_factors'],
      mining_start: '2020-01-01',
      mining_end: '2022-12-31',
      valid_start: '2023-01-01',
      valid_end: '2023-12-31',
    });
    expect(config.headers['Idempotency-Key']).toEqual(expect.any(String));
  });

  it('previews AlphaSAGE from registered capability and keeps validation separate', async () => {
    client.get.mockResolvedValue({ data: { data: { algorithms: [
      { key: 'alphasage', name: 'AlphaSAGE GNN', version: '1.0', valid: true, enabled: true },
    ] } } });

    const result = await previewMining({
      direction: '',
      algorithm: 'alphasage',
      miningStart: '2019-01-01',
      miningEnd: '2022-12-30',
      validStart: '2023-01-03',
      validEnd: '2023-12-29',
      universe: 'csi300',
      alphaSageConfig: { horizon: 5, nEpisodes: 160, poolCapacity: 12, seed: 7 },
    });

    expect(client.get).toHaveBeenCalledWith('/alpha-agent/algorithms');
    expect(client.post).not.toHaveBeenCalled();
    expect(result.data).toMatchObject({
      algorithm: { key: 'alphasage', uses_llm: false },
      window: { mining: ['2019-01-01', '2022-12-30'], valid: ['2023-01-03', '2023-12-29'] },
      alphasage_config: { horizon: 5, nEpisodes: 160, poolCapacity: 12, seed: 7 },
    });
  });

  it('submits AlphaSAGE to the durable queue with its whitelisted GPU parameters', async () => {
    client.post.mockResolvedValue({
      data: { code: 202, data: { task_id: 'alpha-task-1', algorithm: 'alphasage', status: 'queued', queue_position: 1 } },
    });

    const result = await startMining({
      direction: '',
      algorithm: 'alphasage',
      miningStart: '2019-01-01',
      miningEnd: '2022-12-30',
      validStart: '2023-01-03',
      validEnd: '2023-12-29',
      universe: 'csi500',
      alphaSageConfig: { horizon: 10, nEpisodes: 256, poolCapacity: 16, seed: 3 },
    }, 'request-key-alpha');

    const [url, body, config] = client.post.mock.calls[0];
    expect(url).toBe('/alpha-agent/mining/submit');
    expect(body).toEqual({
      algorithm: 'alphasage',
      market: 'a_share',
      freq: 'day',
      universe: 'csi500',
      mining_start: '2019-01-01',
      mining_end: '2022-12-30',
      valid_start: '2023-01-03',
      valid_end: '2023-12-29',
      horizon: 10,
      n_episodes: 256,
      pool_capacity: 16,
      encoder_type: 'gnn',
      seed: 3,
    });
    expect(config.headers['Idempotency-Key']).toBe('request-key-alpha');
    expect(result.data?.taskId).toBe('alpha-task-1');
    expect(result.data?.task.config.algorithm).toBe('alphasage');
  });

  it('previews the exact mining request body without an idempotency or submit side effect', async () => {
    client.post.mockResolvedValue({ data: { code: 200, data: { algorithm: { key: 'rdagent' } } } });
    await previewMining({
      direction: '价量假设',
      algorithm: 'rdagent',
      market: 'a_share',
      universe: 'csi500',
      datasets: ['l1_factors'],
      freq: 'day',
      maxRounds: 4,
      miningStart: '2020-01-01',
      miningEnd: '2022-12-31',
      validStart: '2023-01-01',
      validEnd: '2023-12-31',
    }, 'rd-request-1');
    expect(client.post).toHaveBeenCalledTimes(1);
    const [url, body, options] = client.post.mock.calls[0];
    expect(url).toBe('/alpha-agent/preview');
    expect(body).toMatchObject({
      algorithm: 'rdagent',
      market: 'a_share',
      universe: 'csi500',
      loop_n: 4,
      datasets: ['l1_factors'],
      mining_start: '2020-01-01',
      valid_end: '2023-12-31',
    });
    expect(options).toBeUndefined();
  });
});

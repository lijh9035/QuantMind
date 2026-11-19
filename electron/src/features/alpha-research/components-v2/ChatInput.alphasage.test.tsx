import React from 'react';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const api = vi.hoisted(() => ({ getUniverses: vi.fn(), getMiningAlgorithms: vi.fn() }));
vi.mock('../services-v2/api', () => ({ getUniverses: api.getUniverses }));
vi.mock('../services-v2/algorithmApi', () => ({ getMiningAlgorithms: api.getMiningAlgorithms }));
vi.mock('../../../store', () => ({
  useAppSelector: (selector: (state: unknown) => unknown) => selector({ auth: { isAuthenticated: true, isLoading: false } }),
}));
vi.mock('../../../store/sessionBoundary', () => ({
  getSessionGeneration: () => 1,
  registerSessionReset: () => () => {},
}));
vi.mock('../../../store/sessionScopedStorage', () => ({ getSessionScopedStorageKey: () => null }));

import { ChatInput } from './ChatInput';

const alphaSageCapability = {
  key: 'alphasage',
  name: 'AlphaSAGE GNN',
  version: '1.0',
  enabled: true,
  valid: true,
  markets: ['a_share'],
  freqs: ['day'],
  uses_llm: false,
  uses_datasets: false,
  requires_gpu: true,
};

describe('AlphaSAGE mining input', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.getUniverses.mockResolvedValue({ data: { universes: [{ id: 'csi300', name: '沪深300', stockCount: 300 }] } });
    api.getMiningAlgorithms.mockResolvedValue({ data: { algorithms: [alphaSageCapability], available: true } });
  });

  it('requires both windows and confirms the GPU parameters before submitting', async () => {
    const onPreview = vi.fn().mockResolvedValue({
      algorithm: { key: 'alphasage', name: 'AlphaSAGE GNN', version: '1.0', uses_llm: false, uses_datasets: false },
      market: 'a_share', freq: 'day', universe: 'csi300', data_source: 'qlib_bin',
      loop_n: 1000, direction: '', directions: [], direction_mode: 'selected', datasets: {},
      window: { mode: 'custom', mining: ['2019-01-01', '2022-12-30'], train: ['2019-01-01', '2022-12-30'], valid: ['2023-01-03', '2023-12-29'], oos: ['2023-01-03', '2023-12-29'] },
      requires_llm: false,
      alphasage_config: { horizon: 5, nEpisodes: 1000, poolCapacity: 20, seed: 0 },
    });
    const onSubmit = vi.fn();
    render(<ChatInput inline onPreview={onPreview} onSubmit={onSubmit} />);

    const selectButton = await screen.findByRole('button', { name: 'AlphaSAGE GNN' });
    fireEvent.click(selectButton);
    const startButton = screen.getByRole('button', { name: '预览并开始' });
    expect(startButton).toBeDisabled();
    expect(screen.getByText('独立验证区间（必填）')).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText('AlphaSAGE挖掘开始日期'), { target: { value: '2019-01-01' } });
    fireEvent.change(screen.getByLabelText('AlphaSAGE挖掘结束日期'), { target: { value: '2022-12-30' } });
    fireEvent.change(screen.getByLabelText('AlphaSAGE验证开始日期'), { target: { value: '2023-01-03' } });
    fireEvent.change(screen.getByLabelText('AlphaSAGE验证结束日期'), { target: { value: '2023-12-29' } });
    fireEvent.change(screen.getByLabelText('AlphaSAGE 预测周期'), { target: { value: '10' } });
    fireEvent.change(screen.getByLabelText('AlphaSAGE 训练轮数'), { target: { value: '256' } });
    fireEvent.change(screen.getByLabelText('AlphaSAGE 候选池容量'), { target: { value: '16' } });
    fireEvent.click(startButton);

    await waitFor(() => expect(onPreview).toHaveBeenCalledWith(expect.objectContaining({
      algorithm: 'alphasage',
      miningStart: '2019-01-01', miningEnd: '2022-12-30',
      validStart: '2023-01-03', validEnd: '2023-12-29',
      alphaSageConfig: { horizon: 10, nEpisodes: 256, poolCapacity: 16, seed: 0 },
    })));
    const dialog = await screen.findByRole('dialog', { name: '挖掘参数预览' });
    expect(within(dialog).getByText(/挖掘 2019-01-01 至 2022-12-30；独立验证 2023-01-03 至 2023-12-29/)).toBeInTheDocument();
    fireEvent.click(within(dialog).getByRole('button', { name: '确认并开始' }));
    expect(onSubmit).toHaveBeenCalledWith(expect.objectContaining({
      algorithm: 'alphasage',
      alphaSageConfig: { horizon: 10, nEpisodes: 256, poolCapacity: 16, seed: 0 },
      miningStart: '2019-01-01', validStart: '2023-01-03',
    }));
  });

  it('keeps the statistical OOS window optional', async () => {
    api.getMiningAlgorithms.mockResolvedValue({ data: { algorithms: [{
      key: 'statistical', name: '统计派', version: '1.0', enabled: true, valid: true,
      markets: ['a_share'], freqs: ['day'], uses_llm: false, uses_datasets: false, requires_gpu: false,
    }], available: true } });
    render(<ChatInput inline onPreview={vi.fn()} onSubmit={vi.fn()} />);

    expect(await screen.findByText('样本外检验 OOS（可选）')).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText('统计挖掘开始日期'), { target: { value: '2019-01-01' } });
    fireEvent.change(screen.getByLabelText('统计挖掘结束日期'), { target: { value: '2022-12-30' } });
    expect(screen.getByRole('button', { name: '预览并开始' })).toBeEnabled();
  });
});

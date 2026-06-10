import { afterEach, describe, expect, it, vi } from 'vitest';

const loadConstantsModule = async (env: Record<string, string | undefined> = {}) => {
  vi.resetModules();
  vi.unstubAllEnvs();

  for (const [key, value] of Object.entries(env)) {
    if (value === undefined) {
      delete process.env[key];
    } else {
      vi.stubEnv(key, value);
    }
  }

  return import('@fastgpt/service/common/vectorDB/constants');
};

describe('vectorDB constants', () => {
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.resetModules();
  });

  it('未设置 env 时使用默认数据库名和表名', async () => {
    const { DatasetVectorDbName, DatasetVectorTableName } = await loadConstantsModule({
      DB_NAME: undefined,
      TABLE_NAME: undefined
    });

    expect(DatasetVectorDbName).toBe('fastgpt');
    expect(DatasetVectorTableName).toBe('modeldata');
  });

  it('设置 DB_NAME / TABLE_NAME 时使用配置值', async () => {
    const { DatasetVectorDbName, DatasetVectorTableName } = await loadConstantsModule({
      DB_NAME: 'prod_fastgpt',
      TABLE_NAME: 'prod_vectors'
    });

    expect(DatasetVectorDbName).toBe('prod_fastgpt');
    expect(DatasetVectorTableName).toBe('prod_vectors');
  });

  it('空字符串回退到默认值', async () => {
    const { DatasetVectorDbName, DatasetVectorTableName } = await loadConstantsModule({
      DB_NAME: '   ',
      TABLE_NAME: ''
    });

    expect(DatasetVectorDbName).toBe('fastgpt');
    expect(DatasetVectorTableName).toBe('modeldata');
  });
});

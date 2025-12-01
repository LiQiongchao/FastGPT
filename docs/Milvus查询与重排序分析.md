# Milvus 查询方式与重排序算法分析

## 一、Milvus 查询方式分析

### 1.1 查询入口
**文件位置**: `packages/service/common/vectorStore/milvus/class.ts`

核心查询方法：`embRecall` (第 225-296 行)

### 1.2 查询参数配置

```typescript
const { results } = await client.search({
  collection_name: DatasetVectorTableName,  // 集合名称
  data: vector,                              // 查询向量
  limit,                                     // 返回结果数量限制
  filter: `(teamId == "${teamId}") and (datasetId in [${datasetIds.map((id) => `"${id}"`).join(',')}]) ${collectionIdQuery} ${forbidColQuery}`,
  output_fields: ['collectionId']            // 返回的字段
});
```

### 1.3 索引配置

**索引类型**: HNSW (Hierarchical Navigable Small World)
- **位置**: 第 82-89 行
- **参数配置**:
  ```typescript
  {
    field_name: 'vector',
    index_name: 'vector_HNSW',
    index_type: 'HNSW',
    metric_type: 'IP',           // 内积 (Inner Product)
    params: { 
      efConstruction: 32,        // 构建时的候选数量
      M: 64                      // 每个节点的最大连接数
    }
  }
  ```

### 1.4 查询流程

1. **向量化查询文本**
   - 通过 `getVectorsByText` 将查询文本转换为向量
   - 使用配置的 embedding 模型

2. **构建过滤条件**
   - `teamId`: 团队ID过滤
   - `datasetId`: 数据集ID列表过滤
   - `collectionId`: 集合ID过滤（可选）
   - `forbidCollectionIdList`: 排除的集合ID列表

3. **执行向量搜索**
   - 使用 Milvus SDK 的 `search` 方法
   - 基于 HNSW 索引进行近似最近邻搜索
   - 使用内积 (IP) 作为相似度度量

4. **返回结果**
   - 返回格式：`{ id, collectionId, score }`
   - `score`: 相似度分数（内积值）

### 1.5 查询特点

- **索引算法**: HNSW 图索引，适合大规模向量检索
- **相似度度量**: IP (内积)，值越大表示越相似
- **过滤能力**: 支持多条件组合过滤
- **重试机制**: 失败时自动重试（默认重试2次）

---

## 二、重排序算法逻辑分析

### 2.1 重排序调用流程

**文件位置**: `packages/service/core/dataset/search/controller.ts`

#### 2.1.1 重排序触发条件（第 695-721 行）

```typescript
const reRankResults = await (async () => {
  if (!usingReRank) return [];  // 未启用重排序则跳过
  
  // 1. 合并 embedding 和 fullText 召回结果
  set = new Set<string>(embeddingRecallResults.map((item) => item.id));
  const concatRecallResults = embeddingRecallResults.concat(
    fullTextRecallResults.filter((item) => !set.has(item.id))
  );
  
  // 2. 去重：移除相同的 q 和 a 数据
  set = new Set<string>();
  const filterSameDataResults = concatRecallResults.filter((item) => {
    const str = hashStr(`${item.q}${item.a}`.replace(/[^\p{L}\p{N}]/gu, ''));
    if (set.has(str)) return false;
    set.add(str);
    return true;
  });
  
  // 3. 调用重排序服务
  try {
    return await datasetDataReRank({
      query: reRankQuery,
      data: filterSameDataResults
    });
  } catch (error) {
    usingReRank = false;
    return [];
  }
})();
```

#### 2.1.2 重排序核心函数（第 77-111 行）

```typescript
export const datasetDataReRank = async ({
  data,
  query
}: {
  data: SearchDataResponseItemType[];
  query: string;
}): Promise<SearchDataResponseItemType[]> => {
  // 1. 调用重排序 API
  const results = await reRankRecall({
    query,
    documents: data.map((item) => ({
      id: item.id,
      text: `${item.q}\n${item.a}`  // 将问题和答案拼接作为文档
    }))
  });
  
  // 2. 合并重排序分数到原始数据
  const mergeResult = results
    .map((item, index) => {
      const target = data.find((dataItem) => dataItem.id === item.id);
      if (!target) return null;
      const score = item.score || 0;
      
      return {
        ...target,
        score: [{ type: SearchScoreTypeEnum.reRank, value: score, index }]
      };
    })
    .filter(Boolean) as SearchDataResponseItemType[];
  
  return mergeResult;
};
```

### 2.2 重排序服务实现

**文件位置**: `packages/service/core/ai/rerank/index.ts`

```typescript
export function reRankRecall({
  model = getDefaultRerankModel(),
  query,
  documents
}: {
  model?: ReRankModelItemType;
  query: string;
  documents: { id: string; text: string }[];
}): Promise<ReRankCallResult> {
  // 1. 构建请求
  return POST<PostReRankResponse>(
    model.requestUrl ? model.requestUrl : `${baseUrl}/rerank`,
    {
      model: model.model,                    // 重排序模型名称
      query,                                 // 查询文本
      documents: documents.map((doc) => doc.text)  // 文档列表
    },
    {
      headers: {
        Authorization: model.requestAuth ? `Bearer ${model.requestAuth}` : authorization
      },
      timeout: 30000
    }
  )
  .then((data) => {
    // 2. 处理返回结果
    return data?.results?.map((item) => ({
      id: documents[item.index].id,
      score: item.relevance_score  // 相关性分数
    }));
  });
}
```

### 2.3 重排序模型

**支持的模型**:
- `BAAI/bge-reranker-v2-m3`
- `bge-reranker-v2-m3`
- 其他通过配置添加的重排序模型

**模型特点**:
- 基于 BGE (BAAI General Embedding) 系列
- 使用交叉编码器 (Cross-Encoder) 架构
- 对查询和文档进行深度交互计算相关性分数

### 2.4 RRF 融合算法

**文件位置**: `packages/global/core/dataset/search/utils.ts`

#### 2.4.1 RRF 算法原理

RRF (Reciprocal Rank Fusion) 是一种多结果集融合算法：

```typescript
// RRF 分数计算公式
score = 1 / (k + rank)

其中：
- k: 常数（通常为 60）
- rank: 文档在结果列表中的排名（从1开始）
```

#### 2.4.2 实现逻辑（第 5-72 行）

```typescript
export const datasetSearchResultConcat = (
  arr: { k: number; list: SearchDataResponseItemType[] }[]
): SearchDataResponseItemType[] => {
  const map = new Map<string, SearchDataResponseItemType & { rrfScore: number }>();
  
  // 1. 计算每个文档的 RRF 分数
  arr.forEach((item) => {
    const k = item.k;
    item.list.forEach((data, index) => {
      const rank = index + 1;
      const score = 1 / (k + rank);  // RRF 公式
      
      const record = map.get(data.id);
      if (record) {
        // 2. 合并相同文档的分数（累加 RRF 分数）
        map.set(data.id, {
          ...record,
          rrfScore: record.rrfScore + score
        });
      } else {
        map.set(data.id, {
          ...data,
          rrfScore: score
        });
      }
    });
  });
  
  // 3. 按 RRF 分数排序
  const mapArray = Array.from(map.values());
  const results = mapArray.sort((a, b) => b.rrfScore - a.rrfScore);
  
  return results;
};
```

#### 2.4.3 多结果集融合（第 723-728 行）

```typescript
// 融合三种召回结果
const rrfConcatResults = datasetSearchResultConcat([
  { k: 60, list: embeddingRecallResults },    // 向量召回结果
  { k: 60, list: fullTextRecallResults },      // 全文检索结果
  { k: 58, list: reRankResults }               // 重排序结果（k值稍小，权重更高）
]);
```

**权重说明**:
- `embeddingRecallResults`: k=60
- `fullTextRecallResults`: k=60
- `reRankResults`: k=58（更小的 k 值意味着更高的权重）

### 2.5 完整搜索流程

```
1. 向量召回 (Embedding Recall)
   ↓
2. 全文检索 (Full Text Recall)
   ↓
3. 结果合并与去重
   ↓
4. 重排序 (ReRank) - 可选
   ├─ 调用重排序模型 API
   ├─ 计算查询与文档的相关性分数
   └─ 返回重排序后的结果
   ↓
5. RRF 融合
   ├─ 计算每个文档的 RRF 分数
   ├─ 累加多结果集的 RRF 分数
   └─ 按 RRF 分数排序
   ↓
6. 相似度过滤
   ├─ 如果启用重排序：使用重排序分数过滤
   └─ 否则：使用向量相似度分数过滤
   ↓
7. Token 限制过滤
   └─ 根据最大 Token 限制截断结果
```

### 2.6 重排序的优势

1. **精度提升**: 使用深度模型计算查询与文档的相关性，比简单的向量相似度更准确
2. **语义理解**: 能够理解查询意图和文档内容的深层语义关系
3. **多模态融合**: 结合向量检索、全文检索和重排序，提高召回率和准确率

### 2.7 性能考虑

- **超时设置**: 30秒超时
- **批量处理**: 一次性处理多个文档
- **错误处理**: 重排序失败时降级到不使用重排序
- **去重优化**: 在重排序前先去除重复数据，减少计算量

---

## 三、关键代码位置总结

| 功能 | 文件路径 | 关键方法/行数 |
|------|---------|--------------|
| Milvus 查询 | `packages/service/common/vectorStore/milvus/class.ts` | `embRecall()` (225-296行) |
| 索引配置 | `packages/service/common/vectorStore/milvus/class.ts` | `init()` (35-125行) |
| 重排序调用 | `packages/service/core/dataset/search/controller.ts` | `datasetDataReRank()` (77-111行) |
| 重排序服务 | `packages/service/core/ai/rerank/index.ts` | `reRankRecall()` (16-66行) |
| RRF 算法 | `packages/global/core/dataset/search/utils.ts` | `datasetSearchResultConcat()` (5-72行) |
| 搜索主流程 | `packages/service/core/dataset/search/controller.ts` | `searchDatasetData()` (146-776行) |

---

## 四、总结

### Milvus 查询特点
- ✅ 使用 HNSW 索引，检索效率高
- ✅ 支持多条件过滤
- ✅ 使用内积作为相似度度量
- ✅ 具备自动重试机制

### 重排序算法特点
- ✅ 基于 BGE 交叉编码器模型
- ✅ 使用 RRF 算法融合多结果集
- ✅ 支持多种召回方式的混合排序
- ✅ 具备完善的错误处理和降级机制

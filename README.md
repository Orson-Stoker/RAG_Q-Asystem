## 功能特性

### 1. 文档解析 (`Doc_Parsing/`)
- **多格式支持**：支持常见文档格式的解析
- **OCR 功能**：通过 `ocr.py` 处理扫描文档或图片中的文字
- **智能分块**：使用 `chunking.py` 对文档进行分块
- **文本提取**：`ocr.py`和`chunking.py`分别输出结构化文本 (`document.txt`) 和分块文本 (`doc_chunk.txt`)

### 2. 向量索引 (`Doc_Index/`)
- **本地索引**：基于 OpenSearch 构建本地数据库
- **可配置索引**：通过 `index_config.json` 自定义索引配置
- **索引管理**：提供 `create_index.py` 定义了数据库管理类，负责创建和管理索引以及混合检索

### 3. 智能问答 (`chat.py`)
- **上下文感知**：基于检索到的文档片段进行回答
- **本地&API部署模型运行**：可以调用API或者本地部署模型进行问题回答


## 快速开始
注意：如果要在本地运行需要部署好chat模型放在Chat_local_model文件夹，同时需要用到向量化模型和分词模型在service文件夹中部署，openserch_data文件夹存储了数据
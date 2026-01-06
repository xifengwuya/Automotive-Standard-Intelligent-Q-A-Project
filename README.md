\# 直接运行，自动使用配置中的GPU\_IDS（如\[0,1]两张卡）

python train\_bert.py --train-data ./data/bert\_finetune\_data/train.csv





方式 1：执行评估测试（生成基线对比报告）

bash

运行

python main.py --mode eval

运行完成后，会在 ./evaluations/ 目录生成评估报告，包含各模型的检索指标（Hit Rate@1/3、MRR）和生成指标（BLEU-2、ROUGE-L）对比。

方式 2：启动 FastAPI 接口服务

bash

运行

python main.py --mode api --host 0.0.0.0 --port 8000

接口访问地址：http://localhost:8000/docs，可通过 Swagger UI 进行在线测试。


using Microsoft.ML.Legacy;
using Microsoft.ML.Legacy.Data;
using Microsoft.ML.Legacy.Trainers;
using Microsoft.ML.Legacy.Transforms;
using Microsoft.ML.Runtime.Api;
using System;
using System.IO;

namespace DemoMLDotnetTutorial
{
    /// <summary>
    /// 此程式使用 ML.NET 來學習並預測鳶尾花（Iris）類別
    /// </summary>
    public class Program
    {
        // STEP 1: 定義資料模型

        // IrisData 資料模型用於訓練資料使用，並可作為預測資料模型
        // - 前 4 個屬性為輸入的特性值，用來預測 Label 標籤
        // - Label 標籤是我們要預測的屬性，只有在訓練資料時，才會主動提供值
        public class IrisData
        {
            [Column("0")]
            public float SepalLength;

            [Column("1")]
            public float SepalWidth;

            [Column("2")]
            public float PetalLength;

            [Column("3")]
            public float PetalWidth;

            [Column("4")]
            [ColumnName("Label")]
            public string Label;
        }

        // IrisPrediction 是執行預測後的結果資料模型
        public class IrisPrediction
        {
            [ColumnName("PredictedLabel")]
            public string PredictedLabels;
        }

        public static void Main(string[] args)
        {
            // STEP 2: 建立執行預測運算的 Pipeline
            var pipeline = new LearningPipeline();

            // 訓練預測的數據集的來源位置
            // 若使用 Visual Studio 開發，請確認該檔案有設定"複製到輸出目錄"屬性成"一律複製"
            // 資料集來源：https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
            var dataPath = Path.Combine(Directory.GetCurrentDirectory(), "Data/iris.data.txt");
            // 從數據集中讀取資料，每一行為一筆資料，使用 ',' 當作分隔字元
            pipeline.Add(new TextLoader(dataPath).CreateFrom<IrisData>(separator: ','));

            // STEP 3: 轉換資料
            // 將資料型別為文字的標籤屬性，建立字典並轉透過數字來表示，因為在訓練模型時，只能包含數字型別的值
            pipeline.Add(new Dictionarizer("Label"));

            // 設定要作為學習的特性放入向量中
            pipeline.Add(new ColumnConcatenator("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"));

            // STEP 4: 設定學習器
            // 將要作為學習方法的演算法加入 pipeline 中
            // 這是一個分類的場景，使用 StochasticDualCoordinateAscentClassifier 作為分類方案，預測該鳶尾花是哪種類別
            pipeline.Add(new StochasticDualCoordinateAscentClassifier());

            // 將標籤轉換回原始文字（在 STEP 3 曾將他轉換成數字）
            pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });

            // STEP 5: 根據所提供的數據集來訓練模型
            var model = pipeline.Train<IrisData, IrisPrediction>();

            // STEP 6: 使用訓練後的預測模型進行預測
            // 可以透過改變下列屬性值來測試預測模型
            var prediction1 = model.Predict(new IrisData()
            {
                SepalLength = 3.3f,
                SepalWidth = 1.6f,
                PetalLength = 0.2f,
                PetalWidth = 5.1f,
            });
            Console.WriteLine($"1. 預測的鳶尾花（Iris）類別: {prediction1.PredictedLabels}");

            // ------------------------------

            // STEP 7: 匯出訓練後的預測模型
            // 預測模型的存放位置
            var predictionModelPath = Path.Combine(Directory.GetCurrentDirectory(), "Data/PredictionModel.zip");
            model.WriteAsync(predictionModelPath).ConfigureAwait(false);

            // STEP 8: 載入之前訓練好的預測模型
            var loadPredictionModel = PredictionModel.ReadAsync<IrisData, IrisPrediction>(predictionModelPath).Result;
            
            // STEP 9: 使用匯入的預測模型進行預測
            // 可以透過改變下列屬性值來測試預測模型
            var prediction2 = loadPredictionModel.Predict(new IrisData()
            {
                SepalLength = 5.3f,
                SepalWidth = 4.2f,
                PetalLength = 1.8f,
                PetalWidth = 0.1f,
            });
            Console.WriteLine($"2. 預測的鳶尾花（Iris）類別: {prediction2.PredictedLabels}");
        }
    }
}

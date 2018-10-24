using Microsoft.ML.Legacy;
using Microsoft.ML.Legacy.Data;
using Microsoft.ML.Legacy.Trainers;
using Microsoft.ML.Legacy.Transforms;
using Microsoft.ML.Runtime.Api;
using System;
using System.IO;

namespace DemoMLDotnetTutorial
{
    public class MachineStatusPredictionModel
    {
        // STEP 1: 定義資料模型

        // MachineStatusData 資料模型用於訓練資料使用，並可作為預測資料模型
        // - 前 4 個屬性為輸入的特性值，用來預測 Label 標籤
        // - Label 標籤是我們要預測的屬性，只有在訓練資料時，才會主動提供值
        public class MachineStatusData
        {
            [Column("0")]
            public float MachineTemperature;

            [Column("1")]
            public float MachinePressure;

            [Column("2")]
            public float AmbientTemperature;

            [Column("3")]
            public float AmbientHumidity;

            [Column("4")]
            [ColumnName("Label")]
            public string Label;
        }

        // MachineStatusPrediction 是執行預測後的結果資料模型
        public class MachineStatusPrediction
        {
            [ColumnName("PredictedLabel")]
            public string PredictedLabels;
        }

        public static void Training()
        {
            // STEP 2: 建立執行預測運算的 Pipeline
            var pipeline = new LearningPipeline();

            // 訓練預測的數據集的來源位置
            // 若使用 Visual Studio 開發，請確認該檔案有設定"複製到輸出目錄"屬性成"一律複製"
            var dataPath = Path.Combine(Directory.GetCurrentDirectory(), "Data/MachineStatus.data.csv");
            // 從數據集中讀取資料，每一行為一筆資料，使用 ',' 當作分隔字元
            pipeline.Add(new TextLoader(dataPath).CreateFrom<MachineStatusData>(separator: ','));

            // STEP 3: 轉換資料
            // 將資料型別為文字的標籤屬性，建立字典並轉透過數字來表示，因為在訓練模型時，只能包含數字型別的值
            pipeline.Add(new Dictionarizer("Label"));

            // 設定要作為學習的特性放入向量中
            pipeline.Add(new ColumnConcatenator("Features", "MachineTemperature", "MachinePressure", "AmbientTemperature", "AmbientHumidity"));

            // STEP 4: 設定學習器
            // 將要作為學習方法的演算法加入 pipeline 中
            // 這是一個分類的場景，使用 StochasticDualCoordinateAscentClassifier 作為分類方案，預測該機器狀態是哪種類別
            pipeline.Add(new StochasticDualCoordinateAscentClassifier());

            // 將標籤轉換回原始文字（在 STEP 3 曾將他轉換成數字）
            pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });

            // STEP 5: 根據所提供的數據集來訓練模型
            var model = pipeline.Train<MachineStatusData, MachineStatusPrediction>();

            // STEP 6: 使用訓練後的預測模型進行預測
            // 可以透過改變下列屬性值來測試預測模型
            var prediction1 = model.Predict(new MachineStatusData()
            {
                MachineTemperature = 103.24f,
                MachinePressure = 10.56f,
                AmbientTemperature = 20.31f,
                AmbientHumidity = 25.00f
            });
            Console.WriteLine($"1. 預測的機器狀態（MachineStatus）類別: {prediction1.PredictedLabels}");

            // ------------------------------

            // STEP 7: 匯出訓練後的預測模型
            // 預測模型的存放位置
            var predictionModelPath = Path.Combine(Directory.GetCurrentDirectory(), "Data/MachineStatusPredictionModel.zip");
            model.WriteAsync(predictionModelPath).ConfigureAwait(false);

            // STEP 8: 載入之前訓練好的預測模型
            var loadPredictionModel = PredictionModel.ReadAsync<MachineStatusData, MachineStatusPrediction>(predictionModelPath).Result;

            // STEP 9: 使用匯入的預測模型進行預測
            // 可以透過改變下列屬性值來測試預測模型
            var prediction2 = loadPredictionModel.Predict(new MachineStatusData()
            {
                MachineTemperature = 102.01f,
                MachinePressure = 10.12f,
                AmbientTemperature = 20.41f,
                AmbientHumidity = 24.00f
            });
            Console.WriteLine($"2. 預測的機器狀態（MachineStatus）類別: {prediction2.PredictedLabels}");
        }
    }
}

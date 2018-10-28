using Microsoft.ML.Legacy;
using Microsoft.ML.Legacy.Data;
using Microsoft.ML.Legacy.Trainers;
using Microsoft.ML.Legacy.Transforms;
using Microsoft.ML.Runtime.Api;
using System;
using System.IO;
using System.Threading.Tasks;

namespace DemoMLDotnetTutorial
{
    public class IrisPredictionModelV05
    {
        // STEP 1: 定義資料模型
        //         IrisData 資料模型用於訓練資料使用，並可作為預測資料模型
        //         - 前 4 個屬性為輸入的特性值，用來預測 Label 標籤
        //         - Label 標籤是我們要預測的屬性，只有在訓練資料時，才會主動提供值
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

        // STEP 1: 定義資料模型
        //         IrisPrediction 是執行預測後的結果資料模型
        public class IrisPrediction
        {
            [ColumnName("PredictedLabel")]
            public string PredictedLabel;
        }

        private static string AppPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string DataPath => Path.Combine(AppPath, "Data", "iris-data.txt");
        private static string ModelPath => Path.Combine(AppPath, "Data", "IrisModel.zip");
        private static IrisData DefaultSampleData => new IrisData()
        {
            SepalLength = 3.3f,
            SepalWidth = 1.6f,
            PetalLength = 0.2f,
            PetalWidth = 5.1f,
        };

        public static void Training()
        {
            // STEP 2: 建立執行預測運算的 Pipeline
            var pipeline = new LearningPipeline
            {
                // A. 設定訓練預測的數據集的來源
                //    若使用 Visual Studio 開發，請確認該檔案有設定"複製到輸出目錄"屬性成"一律複製"
                //    資料集來源：https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
                //    從數據集中讀取資料，每一行為一筆資料，使用 ',' 當作分隔字元
                new TextLoader(DataPath).CreateFrom<IrisData>(separator: ','),
                // B. 轉換資料，將資料型別為文字的標籤屬性，建立字典並轉透過數字來表示，因為在訓練模型時，只能包含數字型別的值
                new Dictionarizer("Label"),
                // C. 設定要作為學習的特徵放入向量中
                new ColumnConcatenator("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"),
                // D. 設定學習器，將要作為學習方法的演算法加入 pipeline 中
                //    這是一個分類的場景，使用隨機雙座標上升分類器 (Stochastic Dual Coordinate Ascent Classifier) 作為分類方案，預測該鳶尾花是哪種類別
                new StochasticDualCoordinateAscentClassifier(),
                // E. 轉換資料，將標籤轉換回原始文字（前述步驟曾將他轉換成數字）
                new PredictedLabelColumnOriginalValueConverter(){PredictedLabelColumn = "PredictedLabel"}
            };

            // STEP 3: 根據所提供的數據集來訓練模型
            var model = pipeline.Train<IrisData, IrisPrediction>();

            // STEP 4: 儲存訓練後的預測模型
            model.WriteAsync(ModelPath).ConfigureAwait(false);
        }

        /// <summary>
        /// 使用訓練好的預測模型檔進行預測
        /// </summary>
        /// <param name="sampleData"></param>
        public static async Task PredictWithModelLoadedFromFile(IrisData sampleData = null)
        {
            // 載入之前訓練好的預測模型
            var loadPredictionModel = await PredictionModel.ReadAsync<IrisData, IrisPrediction>(ModelPath);
            // 使用匯入的預測模型進行預測，若無 sampleData 則用預設測試樣本
            var prediction = loadPredictionModel.Predict(sampleData ?? DefaultSampleData);

            Console.WriteLine($"預測的鳶尾花（Iris）類別: {prediction.PredictedLabel}");
        }
    }
}

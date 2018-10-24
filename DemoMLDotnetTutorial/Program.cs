namespace DemoMLDotnetTutorial
{
    /// <summary>
    /// 此程式使用 ML.NET 來訓練並預測模型
    /// </summary>
    public class Program
    {
        public static void Main(string[] args)
        {
            IrisPredictionModel.Training();
            MachineStatusPredictionModel.Training();
        }
    }
}

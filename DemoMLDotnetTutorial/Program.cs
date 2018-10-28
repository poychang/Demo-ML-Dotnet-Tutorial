using System.Threading.Tasks;

namespace DemoMLDotnetTutorial
{
    /// <summary>
    /// 此程式使用 ML.NET 來訓練並預測模型
    /// </summary>
    public class Program
    {
        public static async Task Main(string[] args)
        {
            IrisPredictionModelV05.Training();
            await IrisPredictionModelV05.PredictWithModelLoadedFromFile();
        }
    }
}

from config import FILENAME
from tasks import (DataAggregationTask, DataAnalyzingTask, DataCalculationTask,
                   DataFetchingTask)
from utils import CITIES


def forecast_weather():
    """Анализ погодных условий по городам."""
    cities = list(CITIES.keys())

    fetching = DataFetchingTask()
    fetching.get_data()

    calculation = DataCalculationTask(data=fetching.result)
    calculation.run_concurrent(cities=cities)
    calculation_result = calculation.result

    aggregarion = DataAggregationTask(data=calculation_result)
    aggregarion.process_data()
    aggregarion.save_to_csv(filename=FILENAME)
    aggregarion_result = aggregarion.df

    analyzing = DataAnalyzingTask(df=aggregarion_result)
    best_city = analyzing.analyze_data()
    print(f"Для путешествий {best_city=}")


if __name__ == "__main__":
    forecast_weather()

import concurrent.futures
from queue import Queue
from threading import Thread

import numpy as np
import pandas as pd
from config import (AGGREGATE_WORKERS, BEGIN_HOUR, DAY_HOURS, END_HOUR,
                    FETCHING_WORKERS)
from external.client import YandexWeatherAPI
from external.exceptions import ApiError
from my_logger import logger
from utils import CITIES


class DataFetchingTask:
    """Получение данных от YandexWeatherAPI."""

    def __init__(self):
        self.api = YandexWeatherAPI()
        self.queue = Queue()
        self.result = {}

    def get_weather(self, url) -> dict:
        """Получение данных конкретного города."""
        return self.api.get_forecasting(url)

    def worker(self) -> None:
        """Запуск потока."""
        while True:
            task = self.queue.get()
            city, url = task
            try:
                weather_data = self.get_weather(url)
                self.result[city] = weather_data
            except ApiError as e:
                logger.error("Error in %s: %s", city, str(e))
            finally:
                self.queue.task_done()

    def get_data(self, *args, **kwargs) -> None:
        """Получение данных всех городов."""
        logger.info("Start fetching data")

        workers = FETCHING_WORKERS

        for _ in range(workers):
            thread = Thread(target=self.worker)
            thread.daemon = True
            thread.start()

        for city, url in CITIES.items():
            self.queue.put((city, url))

        self.queue.join()

        logger.info("Finish fetching data")


class DataCalculationTask:
    """Вычисление погодных данных."""

    def __init__(self, data: dict):
        self.all_data = data
        self.result = {}

    def get_city_data(
        self,
        city: str,
        forecast_hours=tuple(range(BEGIN_HOUR, END_HOUR+1)),
    ) -> dict:
        """Вычисление температуры по городу."""
        result = {}
        try:
            city_data = self.all_data[city]
        except KeyError:
            logger.error("Error in %s calculate", city)
            return result

        try:
            forecasts = city_data["forecasts"]
        except KeyError:
            logger.error("%s. No key forecasts", city)
            return result

        for forecast_ in forecasts:
            date = forecast_["date"]
            if len(forecast_["hours"]) < DAY_HOURS:
                continue
            result[date] = []
            for hour_data in forecast_["hours"]:
                if int(hour_data["hour"]) in forecast_hours:
                    result[date].append(
                        {
                            "condition": hour_data["condition"],
                            "temp": hour_data["temp"],
                        }
                    )
        return result

    @staticmethod
    def good_conditions_counter(hours_data: list) -> int:
        """Подсчет количества часов без осадков."""
        right_conditions = ("partly-cloud", "clear", "cloudy", "overcast")
        count = 0
        for hour_data in hours_data:
            if hour_data["condition"] in right_conditions:
                count += 1
        return count

    @staticmethod
    def avg_temp(hours_data: list) -> int | float:
        """Средняя температура."""
        return sum([i["temp"] for i in hours_data]) / len(hours_data)

    def summarize_weather(
        self,
        city: str,
    ) -> dict[str, list[dict[str, int | float]]]:
        """Расчёт погоды по городам."""
        result = []
        try:
            city_data = self.get_city_data(city)
        except KeyError:
            logger.error("Calculate error in %s", city)
            return {}

        for dt, forecast in city_data.items():
            n_hours_with_good_weather = self.good_conditions_counter(
                hours_data=forecast
            )
            avg_temp = self.avg_temp(hours_data=forecast)
            weather_data = {
                "avg_temp": avg_temp,
                "n_hours_good_weather": n_hours_with_good_weather,
            }
            result.append({"date": dt, "weather_data": weather_data})

        return {city: result}

    def run_concurrent(self, cities: list[str]) -> None:
        """Запуск процессного пула."""
        logger.info("Calculate weather data start")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(self.summarize_weather, cities)

        for result in results:
            self.result.update(result)

        logger.info("Calculate weather data finish")


class DataAggregationTask:
    """Вычисление погодных данных."""

    def __init__(self, data: dict):
        self.data = data

    @staticmethod
    def process_chunk(chunk: list) -> pd.DataFrame:
        """Вычисление погодных данных по городам."""
        data = []
        dates = set()

        for city, forecasts in chunk:
            avg_temps = []
            good_weather_hours = []
            for forecast in forecasts:
                date = forecast["date"]
                avg_temp = forecast["weather_data"]["avg_temp"]
                n_hours_with_good_weather = forecast["weather_data"][
                    "n_hours_good_weather"
                ]
                avg_temps.append(avg_temp)
                good_weather_hours.append(n_hours_with_good_weather)
                dates.add(date)
            data.append([city] + avg_temps + good_weather_hours)

        dates = sorted(list(dates))
        columns = (
            ["Город"]
            + [f"Температура, средняя ({date})" for date in dates]
            + [f"Без осадков, часов ({date})" for date in dates]
        )
        return pd.DataFrame(data, columns=columns)

    def process_data(self, workers: int = AGGREGATE_WORKERS) -> None:
        """Вычисление погодных данных."""
        logger.info("Aggregating weather data start")
        items = list(self.data.items())
        chunk_size = len(items) // workers
        chunks = [
            items[i: i + chunk_size] for i in range(0, len(items), chunk_size)
        ]

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
        ) as executor:
            results = list(executor.map(self.process_chunk, chunks))

        merged_results = pd.concat(results, ignore_index=True).fillna("")

        without_precipitation_columns = [
            col for col in merged_results.columns if "Без осадков" in col
        ]
        merged_results["Среднее количество часов без осадков"] = (
            merged_results[without_precipitation_columns]
            .replace("", np.nan)
            .mean(axis=1)
        )
        mean_temp_cols = [
            col for col in merged_results.columns if "Температура" in col
        ]
        merged_results["Средняя Температура"] = (
            merged_results[mean_temp_cols].replace("", np.nan).mean(axis=1)
        )
        self.df = merged_results
        logger.info("Aggregating weather data finish")

    def save_to_csv(self, filename: str) -> None:
        """Сохранение данных в csv."""
        logger.info("Saving data to %s", filename)
        self.df.to_csv(filename, index=False)
        logger.info("Data saved to %s", filename)


class DataAnalyzingTask:
    """Анализ погодных данных."""
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def analyze_data(self) -> str:
        """Анализ погодных данных."""
        logger.info("Analyze weather data start")
        max_avg_temp = self.df["Средняя Температура"].max()

        cities_with_max_temp = self.df[
            self.df["Средняя Температура"] == max_avg_temp
        ]

        if len(cities_with_max_temp) == 1:
            logger.info("Analyze weather data finish")
            return cities_with_max_temp["Город"].tolist()[0]
        max_precipitation_free_days = cities_with_max_temp[
            "Среднее количество часов без осадков"
        ].max()
        best_cities = cities_with_max_temp[
            cities_with_max_temp["Среднее количество часов без осадков"]
            == max_precipitation_free_days
        ]["Город"].tolist()
        logger.info("Analyze weather data finish")
        return ", ".join(best_cities)

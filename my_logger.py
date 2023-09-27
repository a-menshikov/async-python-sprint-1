import logging
import os
from datetime import date

today = date.today().isoformat()

logs_dir = f"./logs/{today}"

if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

log_filename = f"{logs_dir}/forecast.log"

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger = logging.getLogger(__name__)

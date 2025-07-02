#!/usr/bin/env python3
import uuid
from enum import Enum

import orjson

import ipal_iids.settings as settings


class AlertStatus(Enum):
    def __str__(self):
        return str(self.value)

    NEW = "new"
    UPDATE = "updated"
    CLOSED = "closed"


active_alerts = {}
too_many_alerts_warning = False


def _write_alert(alert):
    if settings.alerts:
        if alert["status"] != AlertStatus.UPDATE or settings.alerts_update:
            settings.alertsfd.write(orjson.dumps(alert).decode() + "\n")
            settings.alertsfd.flush()


def raise_alert(ids, point, time, reason, description):
    global too_many_alerts_warning
    key = f"{ids}-{point}"

    if key not in active_alerts:  # Store new alert
        active_alerts[key] = {
            "id": str(uuid.uuid4()),
            "status": AlertStatus.NEW,
            "ids": ids,
            "point": point,
            "reason": reason,
            "description": description,
            "start": time,
            "count": 1,
            "end": None,
        }

    else:  # Update existing alert
        active_alerts[key]["count"] += 1
        active_alerts[key]["status"] = AlertStatus.UPDATE

    _write_alert(active_alerts[key])

    # Purge old alerts that might results in a memory leak over time
    if len(active_alerts) > 1:  # NOTE arbitrary number
        delete = [
            k
            for k in active_alerts
            if time - active_alerts[k]["start"] >= settings.alert_purge_time
        ]

        for key in delete:
            settings.logger.warning(
                f"Purged old alert to save memory: {orjson.dumps(active_alerts[key])}"
            )
            del active_alerts[key]

    # Warn if we accumulated many alerts
    if len(active_alerts) >= 1000:  # NOTE arbitrary number
        if not too_many_alerts_warning:
            settings.logger.warning(
                f"IPAL has stored {len(active_alerts)} active alerts. This can eventually lead to performance issues."
            )
            too_many_alerts_warning = True

    else:
        too_many_alerts_warning = False


def withdraw_alert(ids, point, time):
    key = f"{ids}-{point}"

    if key not in active_alerts:
        return

    active_alerts[key]["end"] = time
    active_alerts[key]["status"] = AlertStatus.CLOSED

    _write_alert(active_alerts[key])

    del active_alerts[key]

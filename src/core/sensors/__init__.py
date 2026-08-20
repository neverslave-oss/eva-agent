"""
sensors package — unified `sensors` tool for EVA.

Reads environmental sensor data (temperature, humidity, soil moisture) from
config-driven Pi endpoints, mirroring the `look` vision tool's structure:

  registry.py — load `sensors` config, resolve ${VAR}, health/status
  pi.py       — client: read_sensors (GET) + set_relay (POST, deferred)
  router.py   — route_sensors(action, registry) + unified envelope
"""

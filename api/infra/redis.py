from redis.asyncio import Redis


def create_redis(url: str) -> Redis:
    return Redis.from_url(
        url,
        decode_responses=False,
        health_check_interval=30,
        socket_connect_timeout=5,
        socket_timeout=5,
    )

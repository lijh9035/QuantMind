"""
股票查询服务FastAPI路由
统一使用FastAPI框架，替换Flask版本
"""

import logging
import math
from datetime import date, timedelta
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import text

from .services import StockQueryService, StockSearchService

logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(prefix="/api/v1", tags=["stock-query"])

# Market-specific stock_daily_latest table mapping
_MARKET_SDL_TABLE = {
    "A": "stock_daily_latest",
    "CN": "stock_daily_latest",
    "HK": "stock_daily_latest_hk",
    "US": "stock_daily_latest_us",
    "CRYPTO": "stock_daily_latest_crypto",
}

# 服务实例
_query_service = None
_search_service = None


def get_query_service() -> StockQueryService:
    """获取股票查询服务实例"""
    global _query_service
    if _query_service is None:
        _query_service = StockQueryService()
    return _query_service


def get_search_service() -> StockSearchService:
    """获取股票搜索服务实例"""
    global _search_service
    if _search_service is None:
        _search_service = StockSearchService()
    return _search_service

    # 请求模型


class StockQueryRequest(BaseModel):
    symbol: str
    fields: list[str] | None = None


class StockSearchRequest(BaseModel):
    query: str
    limit: int | None = 10


class MarketIndexRequest(BaseModel):
    symbols: list[str] | None = None
    indicators: list[str] | None = None


@router.get("/stocks/search")
async def search_stocks(
    q: str = Query(..., description="搜索关键词"),
    limit: int = Query(10, ge=1, le=100, description="返回结果数量限制"),
):
    """搜索股票"""
    logger.info("Stock search request", extra={"query": q, "limit": limit})

    try:
        search_service = get_search_service()
        results = await search_service.search_stocks(q, limit)

        logger.info(
            "Stock search completed",
            extra={"query": q, "results_count": len(results), "limit": limit},
        )

        return {
            "query": q,
            "results": results,
            "total": len(results),
            "timestamp": __import__("datetime").datetime.now().isoformat() + "Z",
        }
    except Exception as e:
        logger.error("Stock search failed", extra={"query": q, "error": str(e)})
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")


@router.get("/stocks/all")
async def get_all_stocks(
    market: str = Query("A", description="市场代码: A/CN, HK, US, CRYPTO"),
    exchange: str = Query("", description="交易所筛选: SH, SZ, BJ (仅A股有效)"),
    enrich: bool = Query(False, description="是否包含 PE/PB/市值等字段"),
    limit: int = Query(10000, ge=1, le=50000, description="返回数量限制"),
):
    """获取指定市场的全部股票列表，A股支持按交易所筛选"""
    market_key = market.upper()
    table = _MARKET_SDL_TABLE.get(market_key, "stock_daily_latest")
    exchange_key = exchange.upper().strip()
    logger.info("Fetching all stocks", extra={"market": market_key, "exchange": exchange_key, "table": table, "enrich": enrich})

    try:
        from backend.shared.database_pool import get_db

        # Determine name column: CN uses stock_name, others use name
        is_cn = market_key in ("A", "CN")
        name_col = "stock_name" if is_cn else "name"

        # Build exchange filter for A-share (symbol prefix: SH, SZ, BJ)
        exchange_where = ""
        exchange_params: dict = {"limit": limit}
        is_cn = market_key in ("A", "CN")
        if is_cn and exchange_key in ("SH", "SZ", "BJ"):
            exchange_where = "WHERE symbol LIKE :exchange_prefix"
            exchange_params["exchange_prefix"] = f"{exchange_key}%"

        if enrich:
            name_col = "stock_name" if is_cn else "name"
            sql = f"""
                SELECT symbol,
                       COALESCE({name_col}, '') AS name,
                       COALESCE(close, 0) AS close,
                       COALESCE(pe_ttm, 0) AS pe,
                       COALESCE(pb, 0) AS pb,
                       COALESCE(roe, 0) AS roe,
                       COALESCE(total_mv, 0) AS "marketCap",
                       COALESCE(float_mv, 0) AS "floatMarketCap",
                       COALESCE(turnover_rate, 0) AS "turnoverRate",
                       COALESCE(pct_change, 0) AS "pctChange",
                       COALESCE(is_st, 0) <> 0 AS "isSt"
                FROM {table}
                {exchange_where}
                ORDER BY symbol
                LIMIT :limit
            """
        else:
            name_col = "stock_name" if is_cn else "name"
            sql = f"""
                SELECT symbol, COALESCE({name_col}, '') AS name
                FROM {table}
                {exchange_where}
                ORDER BY symbol
                LIMIT :limit
            """

        with get_db() as db:
            rows = db.execute(text(sql), exchange_params).fetchall()

        items = []
        for row in rows:
            item = {"symbol": row[0], "name": row[1]}
            if enrich:
                item.update({
                    "close": float(row[2] or 0),
                    "pe": float(row[3] or 0) if row[3] else None,
                    "pb": float(row[4] or 0) if row[4] else None,
                    "roe": float(row[5] or 0) if row[5] else None,
                    "marketCap": float(row[6] or 0) if row[6] else None,
                    "floatMarketCap": float(row[7] or 0) if row[7] else None,
                    "turnoverRate": float(row[8] or 0) if row[8] else None,
                    "pctChange": float(row[9] or 0) if row[9] else None,
                    "isSt": bool(row[10]) if len(row) > 10 else False,
                })
            items.append(item)

        return {
            "items": items,
            "total": len(items),
            "market": market_key,
            "exchange": exchange_key or None,
            "table": table,
        }
    except Exception as e:
        logger.error("Get all stocks failed", extra={"market": market_key, "error": str(e)})
        raise HTTPException(status_code=500, detail=f"获取股票列表失败: {str(e)}")


@router.get("/stocks/{symbol}")
async def get_stock_info(symbol: str):
    """获取股票详细信息"""
    logger.info("Fetching stock info", extra={"symbol": symbol})

    try:
        query_service = get_query_service()
        stock_info_resp = await query_service.get_stock_info(symbol)
        if not stock_info_resp or not bool(getattr(stock_info_resp, "success", False)):
            raise HTTPException(status_code=500, detail=f"获取股票信息失败: {getattr(stock_info_resp, 'message', 'unknown error')}")

        stock_info = getattr(stock_info_resp, "data", None)
        if not isinstance(stock_info, dict):
            logger.warning("Stock not found", extra={"symbol": symbol})
            raise HTTPException(status_code=404, detail=f"股票 {symbol} 未找到")

        logger.info(
            "Stock info retrieved successfully",
            extra={
                "symbol": symbol,
                "stock_name": stock_info.get("name", ""),
                "price": stock_info.get("price", 0),
            },
        )

        return stock_info_resp.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Get stock info failed", extra={"symbol": symbol, "error": str(e)})
        raise HTTPException(status_code=500, detail=f"获取股票信息失败: {str(e)}")


@router.get("/stocks/{symbol}/quote")
async def get_stock_quote(symbol: str):
    """获取股票实时报价"""
    logger.info("Fetching stock quote", extra={"symbol": symbol})

    try:
        query_service = get_query_service()
        quote = await query_service.get_stock_quote(symbol)

        if not quote:
            logger.warning("Quote not found", extra={"symbol": symbol})
            raise HTTPException(status_code=404, detail=f"股票 {symbol} 报价未找到")

        logger.info(
            "Stock quote retrieved successfully",
            extra={
                "symbol": symbol,
                "price": quote.get("price", 0),
                "change_percent": quote.get("change_percent", 0),
            },
        )

        return quote
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Get stock quote failed", extra={"symbol": symbol, "error": str(e)})
        raise HTTPException(status_code=500, detail=f"获取股票报价失败: {str(e)}")


@router.get("/market/indices")
async def get_market_indices(
    symbols: list[str] | None = Query(None, description="指数代码列表"),
    indicators: list[str] | None = Query(None, description="技术指标列表"),
):
    """获取大盘指数信息"""
    logger.info("Fetching market indices", extra={"symbols": symbols, "indicators": indicators})

    try:
        query_service = get_query_service()
        indices = query_service.get_market_indices(symbols, indicators)

        logger.info(
            "Market indices retrieved successfully",
            extra={"count": len(indices)},
        )

        return {
            "indices": indices,
            "total": len(indices),
            "timestamp": __import__("datetime").datetime.now().isoformat() + "Z",
        }
    except Exception as e:
        logger.error("Get market indices failed", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=f"获取大盘指数失败: {str(e)}")


@router.get("/market/overview")
async def get_market_overview():
    """获取本地 QuantDB 的真实市场概览（最近可用交易日）。"""
    logger.info("Fetching market overview")

    try:
        from backend.services.engine.data_platform.quantdb_hub import QuantDBDataHub

        hub = QuantDBDataHub.get_instance()
        end = date.today()
        start = end - timedelta(days=14)
        index_meta = [
            ("000001.SH", "上证指数"),
            ("399001.SZ", "深证成指"),
            ("000300.SH", "沪深300"),
            ("000905.SH", "中证500"),
            ("399006.SZ", "创业板指"),
            ("000688.SH", "科创50"),
            ("000016.SH", "上证50"),
            ("899050.BJ", "北证50"),
        ]

        def _number(value, default=0.0):
            try:
                result = float(value)
                return result if math.isfinite(result) else default
            except (TypeError, ValueError):
                return default

        indices = []
        for symbol, name in index_meta:
            try:
                frame = hub.fetch_index_kline(symbol, start, end)
                if frame is None or frame.empty:
                    continue
                frame = frame.sort_values("trade_date").reset_index(drop=True)
                latest = frame.iloc[-1]
                previous = frame.iloc[-2] if len(frame) >= 2 else None

                price = _number(latest.get("close"))
                if price <= 0:
                    continue
                previous_close = _number(previous.get("close")) if previous is not None else 0.0
                if previous_close <= 0:
                    previous_close = _number(latest.get("open"), price) or price
                change = price - previous_close
                trade_date = str(latest.get("trade_date") or "")[:10]

                indices.append({
                    "symbol": symbol,
                    "name": name,
                    "price": round(price, 4),
                    "change": round(change, 4),
                    "change_percent": round(change / previous_close * 100, 4) if previous_close else 0.0,
                    "open": round(_number(latest.get("open")), 4),
                    "high": round(_number(latest.get("high")), 4),
                    "low": round(_number(latest.get("low")), 4),
                    "pre_close": round(previous_close, 4),
                    "volume": _number(latest.get("volume")),
                    "amount": _number(latest.get("amount")),
                    "trade_date": trade_date,
                })
            except Exception as index_error:
                logger.warning(
                    "Skipping unavailable market index",
                    extra={"symbol": symbol, "error": str(index_error)},
                )

        if not indices:
            return {
                "success": False,
                "data": {"market": "CN", "indices": [], "last_update": "", "source_used": "local_quantdb"},
                "error": "CN 市场暂无可用的本地指数行情",
            }

        up = sum(1 for item in indices if item["change_percent"] > 0)
        down = sum(1 for item in indices if item["change_percent"] < 0)
        return {
            "success": True,
            "data": {
                "market": "CN",
                "indices": indices,
                "last_update": max(item["trade_date"] for item in indices if item["trade_date"]),
                "stats": {"up": up, "down": down, "flat": len(indices) - up - down, "total": len(indices)},
                "source_used": "local_quantdb",
            },
        }
    except Exception as exc:
        logger.error("Get market overview failed", extra={"error": str(exc)})
        raise HTTPException(status_code=500, detail=f"获取市场概览失败: {str(exc)}")



"""
注意：企业级金融业务禁止内置任何演示数据接口。
如需数据，请通过真实数据源与本地数据库/数据管理链路提供。
"""

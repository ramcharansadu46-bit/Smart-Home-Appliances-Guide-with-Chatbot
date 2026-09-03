from uuid import uuid4

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
import os
import logging
import re
from typing import Dict, List, Tuple, Optional

import uvicorn

from .config import SEARCH_MODE, GROQ_API_KEY, SERPAPI_KEY, SESSION_SECRET_KEY, SESSION_COOKIE_SECURE
from .ai_service import analyze_text_with_groq, build_search_query, generate_product_reason, compute_match_score, generate_product_analysis
from .shopping_service import serpapi_search, rank_products, filter_relevant_products
from .models import AdvisorResponse, Product, CompareResponse, ParsedUserIntent, AIAnalysis
from .keywords import PRODUCT_CATEGORIES, SMART_HOME_BRANDS, FEATURE_KEYWORDS, PRICE_RANGES, ALL_KEYWORDS
from .auth import auth_router, GUEST_COOKIE_NAME, ensure_identity
from .database import (
    init_db,
    save_chat_entry,
    get_recent_history,
    get_recent_queries_for_context,
    get_history_item,
)

app = FastAPI(
    title="AI Smart Home Advisor API",
    description="Academic Hybrid AI Architecture (Search, ML, NLP)",
    version="3.1.0"
)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET_KEY,
    https_only=SESSION_COOKIE_SECURE,
    same_site="lax",
)
app.include_router(auth_router)

SMART_HOME_QUERY_MESSAGE = (
    "Please search for smart home products only. "
    "This query appears unrelated to smart home appliances."
)

UNSAFE_CONTENT_MESSAGE = (
    "I can only help with smart home products. Please try a home appliance or automation query."
)

UNSAFE_CONTENT_PATTERNS = [
    r"\bsex\s*toy\b",
    r"\badult\s*toy\b",
    r"\bintimate\s*toy\b",
    r"\bvibrator\b",
    r"\bdildo\b",
    r"\bporn\b",
    r"\berotic\b",
    r"\bnsfw\b",
    r"\bmasturbat(?:e|ion|or|ing)?\b",
    r"\bpleasure\b",
]

# Setup templates 
APP_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(APP_DIR, "templates"))
templates.env.cache = {}

# Mount static files
STATIC_DIR = os.path.join(APP_DIR, "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
async def on_startup() -> None:
    init_db()


@app.middleware("http")
async def guest_identity_middleware(request: Request, call_next):
    request.state.guest_id = request.cookies.get(GUEST_COOKIE_NAME) or str(uuid4())
    response = await call_next(request)

    session_user = request.session.get("user") if hasattr(request, "session") else None
    existing_guest_cookie = request.cookies.get(GUEST_COOKIE_NAME)

    if session_user and existing_guest_cookie:
        response.delete_cookie(GUEST_COOKIE_NAME)
    elif not session_user and not existing_guest_cookie:
        response.set_cookie(
            key=GUEST_COOKIE_NAME,
            value=request.state.guest_id,
            httponly=True,
            secure=SESSION_COOKIE_SECURE,
            samesite="lax",
            max_age=60 * 60 * 24 * 30,
        )
    return response

def _build_smart_home_hints() -> set:
    hints = {
        "smart", "home", "automation", "iot", "wifi", "zigbee", "z-wave", "matter", "thread",
        "alexa", "google", "homekit", "siri", "security", "camera", "doorbell", "lock", "sensor",
        "bulb", "light", "switch", "plug", "speaker", "display", "thermostat", "purifier", "fan",
        "vacuum", "air", "kitchen", "tv", "router", "mesh", "watch", "tracker", "appliance",
    }
    for keyword in ALL_KEYWORDS:
        kw = str(keyword).strip().lower()
        if kw:
            hints.add(kw)
            hints.update(re.findall(r"[a-z0-9]+", kw))
    return hints

SMART_HOME_HINTS = _build_smart_home_hints()

def _is_smart_home_related(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q: return False
    tokens = re.findall(r"[a-z0-9]+", q)
    overlap = sum(1 for token in tokens if token in SMART_HOME_HINTS)
    return overlap >= 1


def _contains_unsafe_content(text: str) -> bool:
    q = (text or "").strip().lower()
    if not q:
        return False
    return any(re.search(pattern, q, re.IGNORECASE) for pattern in UNSAFE_CONTENT_PATTERNS)

def _score_products_for_comparison(products: List[Product]) -> Tuple[Product, str, List[Dict]]:
    sorted_p = sorted(products, key=lambda x: x.match_score or 0, reverse=True)
    best = sorted_p[0]
    return best, f"{best.name} is the best overall buy based on heuristic alignment.", []


def _build_rank_requirements(ai_result: AIAnalysis, budget_override: Optional[int] = None) -> Dict:
    budget_range = dict(ai_result.budget_range or {})
    if budget_override is not None and budget_override > 0:
        budget_range["max"] = budget_override
    return {
        "required_features": ai_result.required_features or [],
        "category": ai_result.product_category or "",
        "budget_range": budget_range,
    }


def _build_reason_from_ai(product: Dict, ai_result: AIAnalysis) -> str:
    name = product.get("name", "Unknown")
    features = [str(feature).strip() for feature in product.get("features", []) if str(feature).strip()]
    category = (ai_result.product_category or product.get("category") or "smart home product").strip()

    if ai_result.ai_explanation:
        explanation = ai_result.ai_explanation.strip().rstrip(".")
        feature_hint = ", ".join(features[:2]) if features else "the requested feature set"
        return f"{name} is recommended because {explanation}. It also matches the product profile with {feature_hint}."

    if ai_result.buying_advice:
        advice = ai_result.buying_advice.strip().rstrip(".")
        feature_hint = ", ".join(features[:2]) if features else "the expected use case"
        return f"{name} is recommended because {advice}. It aligns with {feature_hint} for {category.lower()}."

    if features:
        return f"{name} is a practical match for {category.lower()} use because it includes {', '.join(features[:2])}."

    return f"{name} is a balanced option for {category.lower()} use and the available price-to-feature mix."


def _build_serpapi_context(product: Dict) -> str:
    parts = [
        f"title: {product.get('raw_title') or product.get('name') or ''}",
        f"store: {product.get('source') or product.get('store') or ''}",
        f"price: {product.get('price_text') or product.get('price_inr') or ''}",
        f"rating: {product.get('rating', 0)}",
        f"reviews: {product.get('review_count', 0)}",
        f"features: {', '.join(product.get('features', []))}",
        f"snippet: {product.get('snippet') or ''}",
        f"extensions: {', '.join(product.get('extensions', [])) if isinstance(product.get('extensions', []), list) else product.get('extensions', '')}",
        f"delivery: {product.get('delivery') or ''}",
    ]
    return " | ".join(part for part in parts if part and part.split(": ", 1)[-1].strip())


def _normalize_bullets(items: List[str]) -> List[str]:
    normalized = []
    for item in items or []:
        text = re.sub(r"\s+", " ", str(item).strip())
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def _build_history_summary(recommended: List[Product], ai_result: AIAnalysis) -> str:
    category = (ai_result.product_category or "Smart Appliance").strip()
    if not recommended:
        return f"No strong {category.lower()} matches found in this run."
    top = recommended[0]
    return f"Top pick: {top.name} ({int(top.match_score or 0)}% match) in {category}."


def _looks_generic_bullet(text: str) -> bool:
    generic_patterns = [
        r"\bgood product\b",
        r"\bnice design\b",
        r"\benhanced home automation\b",
        r"\bsmart home integration\b",
        r"\buser[- ]friendly\b",
        r"\bstrong rating\b",
        r"\bvalue for money\b",
        r"\bcommon smart features\b",
        r"\bpractical choice\b",
    ]
    lower = (text or "").lower()
    return any(re.search(pattern, lower) for pattern in generic_patterns)


def _grounded_pros_cons(product: Dict, ai_result: AIAnalysis) -> Tuple[List[str], List[str]]:
    name = str(product.get("name", "Unknown")).strip()
    raw_title = str(product.get("raw_title") or name).strip()
    price = int(product.get("price_inr", 0) or 0)
    rating = float(product.get("rating", 0) or 0)
    review_count = int(product.get("review_count", 0) or 0)
    features = [str(feature).strip() for feature in product.get("features", []) if str(feature).strip()]
    snippet = str(product.get("snippet", "")).strip()
    delivery = str(product.get("delivery", product.get("shipping", ""))).strip()
    extensions = product.get("extensions", [])
    if not isinstance(extensions, list):
        extensions = [str(extensions)] if str(extensions).strip() else []
    extension_text = ", ".join(str(ext).strip() for ext in extensions if str(ext).strip())
    store = str(product.get("source") or product.get("store") or "").strip()
    price_text = str(product.get("price_text") or f"₹{price}" if price else "").strip()

    text_blob = " ".join([name, raw_title, snippet, delivery, extension_text, store, " ".join(features)]).lower()

    pros: List[str] = []
    cons: List[str] = []

    def add_pro(value: str) -> None:
        if value and value not in pros:
            pros.append(value)

    def add_con(value: str) -> None:
        if value and value not in cons:
            cons.append(value)

    if features:
        add_pro(f"Listing explicitly mentions {', '.join(features[:2])}.")
    if snippet:
        first_clause = re.split(r"[.;|]", snippet)[0].strip()
        if first_clause:
            add_pro(f"Listing details mention: {first_clause}.")
    if extension_text:
        add_pro(f"Marketplace listing shows extras such as {extension_text}.")
    if delivery:
        add_pro(f"Delivery information is visible: {delivery}.")
    if store:
        add_pro(f"Sold via {store}, so the merchant is clearly identified.")
    if rating >= 4.4:
        add_pro(f"Strong customer satisfaction at {rating:.1f}/5.")
    elif rating >= 3.8:
        add_pro(f"Decent customer rating at {rating:.1f}/5.")

    if price:
        if price <= 3000:
            add_pro("Lower entry price makes it easier to try the category.")
        elif price >= 20000:
            add_con("Premium pricing means it is a serious purchase, not a casual add-on.")
        else:
            add_pro("Price sits in a middle band for this product class.")

    if review_count >= 1000:
        add_pro(f"High review volume ({review_count} reviews) gives the listing more buyer signal.")
    elif review_count > 0:
        add_pro(f"Review count is available at {review_count}, which adds some purchase signal.")
    else:
        add_con("The listing does not show much review history.")

    if "wifi" in text_blob or "app" in text_blob or "smart" in text_blob:
        add_pro("Connected control is explicitly supported in the listing text.")
    if "inverter" in text_blob:
        add_pro("Inverter support suggests better efficiency for continuous use.")
    if "battery" in text_blob:
        add_pro("Battery-powered operation helps when wiring is inconvenient.")

    if price and price >= 15000 and ai_result.product_category and ai_result.product_category.lower() not in text_blob:
        add_con("The price is high relative to the amount of listing detail shown.")
    if not snippet:
        add_con("The listing snippet is sparse, so there is less real-world detail to judge from.")
    if not extension_text:
        add_con("No extra merchant benefits or bundles are shown in the listing.")
    if rating and rating < 4.0:
        add_con(f"Rating is only {rating:.1f}/5, which suggests mixed buyer feedback.")
    if review_count and review_count < 100:
        add_con("Low review volume makes the signal less reliable.")

    if not pros:
        title_excerpt = " ".join(raw_title.split()[:6]).strip()
        add_pro(f"Listing title is {title_excerpt}.")
    if not cons:
        add_con("The listing does not show enough distinct drawbacks to judge.")

    return _normalize_bullets(pros)[:4], _normalize_bullets(cons)[:4]


async def generate_pros_cons(product: Dict, ai_result: AIAnalysis) -> Tuple[List[str], List[str]]:
    return _grounded_pros_cons(product, ai_result)

# ─── Home Page ────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the home page."""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "categories": list(PRODUCT_CATEGORIES.keys()),
        },
    )


@app.get("/how-it-works", response_class=HTMLResponse)
async def how_it_works(request: Request):
    """Serve the How It Works page."""
    return templates.TemplateResponse("how_it_works.html", {
        "request": request,
    })

# ─── AI Advisor (v3.1.0 Hybrid Architecture) ──────────────────
@app.post("/advisor")
async def advisor(request: Request):
    """
    Hybrid Flow:
    1. Extract Intent (Groq/NLP)
    2. Search (SerpAPI)
    3. Heuristic Ranking (Python/A*)
    4. Sentiment Analysis (Python/NLP)
    5. Final Reasoning (Groq/CoT)
    """
    data = {}
    if request.headers.get("content-type", "").startswith("application/json"):
        data = await request.json()
    else:
        form = await request.form()
        data = dict(form)

    user_text = (data.get("query") or data.get("text") or "").strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Empty query is not allowed")
    if _contains_unsafe_content(user_text):
        raise HTTPException(status_code=400, detail=UNSAFE_CONTENT_MESSAGE)
    if not _is_smart_home_related(user_text):
        raise HTTPException(status_code=400, detail=SMART_HOME_QUERY_MESSAGE)

    identity = ensure_identity(request)
    recent_queries = get_recent_queries_for_context(identity.get("user_id"), identity.get("guest_id"), limit=3)

    # PAGE 1: NLP Intent Extraction (Groq)
    try:
        ai_result = await analyze_text_with_groq(user_text, previous_queries=recent_queries)
        if ai_result is None:
            raise HTTPException(status_code=502, detail="Intent extraction failed")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during AI analysis: {e}")
        # Provide fallback on any unexpected error
        ai_result = AIAnalysis(
            product_category="Smart Appliance",
            required_features=["Smart", "WiFi"],
            brand=None,
            budget_range={"min": 0, "max": 999999},
            usage_scenario="Home automation",
            buying_advice="Looking for smart home devices.",
            ai_explanation="Smart appliances for your home."
        )

    # PAGE 2: Search Space Generation (SerpAPI)
    search_query = build_search_query(ai_result)
    min_p, max_p = None, None
    if ai_result.budget_range:
        min_p = ai_result.budget_range.get("min")
        max_p = ai_result.budget_range.get("max")
    
    raw_products = await serpapi_search(search_query, max_results=20, min_price=min_p, max_price=max_p)
    if not raw_products:
        raw_products = []

    filtered_products = filter_relevant_products(raw_products, ai_result, user_text)
    if filtered_products:
        raw_products = filtered_products

    # PAGE 3: Local Heuristic Search & Sentiment Analysis (Python)
    requirements = _build_rank_requirements(ai_result)
    try:
        ranked_results = rank_products(raw_products, requirements)
        if not ranked_results:
            ranked_results = []
    except Exception as e:
        logger.error(f"Error during ranking: {e}")
        # Fallback: return products with simple scoring
        ranked_results = [
            {
                **p,
                "match_score": 85 - (min(i * 5, 50)),
                "reason": _build_reason_from_ai(p, ai_result)
            }
            for i, p in enumerate(raw_products)
        ]

    # PAGE 4: Mapping to modern UI/UX Response Structure
    recommended = []
    for p in ranked_results:
        pros, cons = await generate_pros_cons(p, ai_result)
        prod = Product(
            name=p.get("name", "Unknown"),
            price_inr=int(p.get("price_inr", 0) or 0),
            rating=float(p.get("rating", 3.5) or 3.5),
            review_count=int(p.get("review_count", 0)),
            image_url=p.get("image_url", ""),
            product_link=p.get("product_link", ""),
            store=p.get("store", ""),
            features=p.get("features", []),
            match_score=float(p.get("match_score", 0)),
            ai_reason=p.get("reason") or _build_reason_from_ai(p, ai_result),
            pros=pros,
            cons=cons,
        )
        recommended.append(prod)

    # Map to the new intent structure expected by scripts.js
    intent = ParsedUserIntent(
        category=ai_result.product_category or "Any",
        budget=ai_result.budget_range.get("max") if ai_result.budget_range else "Flexible",
        room_size=ai_result.usage_scenario or "Any",
        preferences=ai_result.required_features,
        energy_efficiency="High Efficiency" if "star" in user_text.lower() else "Standard"
    )

    best_overall_product = recommended[0] if recommended else None
    best_overall_reason = f"Based on Heuristic Match Score of {best_overall_product.match_score}%, this is our top expert recommendation." if best_overall_product else ""

    response_payload = AdvisorResponse(
        ai_analysis=intent,
        recommended_products=recommended,
        best_overall_product=best_overall_product,
        best_overall_reason=best_overall_reason,
        follow_up_question=ai_result.buying_advice or "Would you like to see more premium options?"
    ).dict()

    summary = _build_history_summary(recommended, ai_result)
    save_chat_entry(
        query=user_text,
        response_payload=response_payload,
        summary=summary,
        user_id=identity.get("user_id"),
        guest_id=identity.get("guest_id") if not identity.get("user_id") else None,
    )

    return response_payload


# ─── Manual Search ────────────────────────────────────────────
@app.post("/search/manual")
async def manual_search(request: Request):
    data = {}
    if request.headers.get("content-type", "").startswith("application/json"):
        data = await request.json()
    else:
        form = await request.form()
        data = dict(form)

    query = data.get("query", "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    if _contains_unsafe_content(query):
        raise HTTPException(status_code=400, detail=UNSAFE_CONTENT_MESSAGE)

    min_price = data.get("min_price")
    max_price = data.get("max_price")
    category = data.get("category", "").strip()

    if not _is_smart_home_related(f"{category} {query}".strip()):
        raise HTTPException(status_code=400, detail=SMART_HOME_QUERY_MESSAGE)

    try:
        min_price = int(min_price) if min_price else None
        max_price = int(max_price) if max_price else None
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid price format")

    search_query = f"{category} {query}" if category else query
    products_data = await serpapi_search(search_query, max_results=20, min_price=min_price, max_price=max_price)

    manual_ai_result = AIAnalysis(product_category=category or "Smart Appliance", required_features=[])
    products_data = filter_relevant_products(products_data or [], manual_ai_result, query)

    if not products_data:
        return {
            "query": query,
            "filters": {"min_price": min_price, "max_price": max_price, "category": category},
            "products": [],
            "total_results": 0,
            "message": "No products found. Try different keywords."
        }

    products = []
    for p in products_data:
        try:
            pros, cons = await generate_pros_cons(p, manual_ai_result)
            prod = Product(
                name=p.get("name", "Unknown"),
                price_inr=int(p.get("price_inr", 0) or 0),
                rating=float(p.get("rating", 3.5) or 3.5),
                review_count=int(p.get("review_count", 0) or 0),
                image_url=p.get("image_url"),
                product_link=p.get("product_link"),
                store=p.get("store", ""),
                features=p.get("features", []),
                match_score=float(p.get("match_score", 0)),
                ai_reason=p.get("reason") or _build_reason_from_ai(p, manual_ai_result),
                pros=pros,
                cons=cons,
            )
            products.append(prod)
        except Exception:
            logger.exception("Skipping invalid product")

    best_overall_product = None
    best_overall_reason = None
    if products:
        best_overall_product, best_overall_reason, _ = _score_products_for_comparison(products)

    return {
        "query": query,
        "filters": {"min_price": min_price, "max_price": max_price, "category": category},
        "products": [p.dict() for p in products],
        "total_results": len(products),
        "best_overall_product": best_overall_product.dict() if best_overall_product else None,
        "best_overall_reason": best_overall_reason,
    }


# ─── Product Compare ─────────────────────────────────────────
@app.post("/compare")
async def compare_products(request: Request):
    """Compare 2-6 products side by side."""
    data = await request.json()
    products_data = data.get("products", [])

    if len(products_data) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 products to compare")
    if len(products_data) > 6:
        products_data = products_data[:6]

    products = []
    for p in products_data:
        try:
            prod = Product(**p)
            products.append(prod)
        except Exception:
            logger.exception("Skipping invalid product in compare")

    if len(products) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 valid products")

    # Use new Advanced AI Comparison
    from .ai_service import generate_advanced_comparison
    # Prepare products as dicts for the AI service
    products_dicts = [p.dict() for p in products]
    
    advanced_comparison = await generate_advanced_comparison(products_dicts)

    if advanced_comparison:
        # Match back to get the best product object
        best_product = next((p for p in products if p.name.lower() == advanced_comparison.best_product_name.lower()), products[0])
        
        # Enrich the product objects with the AI analysis (pros/cons/verdict)
        for p in products:
            analysis = next((a for a in advanced_comparison.side_by_side_analysis if a.name.lower() == p.name.lower()), None)
            if analysis:
                p.pros = analysis.pros
                p.cons = analysis.cons
                p.ai_reason = analysis.verdict

        return {
            "products": [p.dict() for p in products],
            "comparison_summary": advanced_comparison.comparison_summary,
            "best_product": best_product.dict(),
            "best_product_reason": advanced_comparison.best_product_reason,
            "overall_recommendation": advanced_comparison.overall_recommendation
        }

    # Fallback if AI comparison fails
    best_product, best_reason, score_breakdown = _score_products_for_comparison(products)
    summary = f"Best product to buy: {best_product.name}. Note: Advanced AI comparison is currently unavailable, using fallback scoring."

    return CompareResponse(
        products=products,
        comparison_summary=summary,
        best_product=best_product,
        best_product_reason=best_reason,
        score_breakdown=score_breakdown,
    ).dict()



# ─── API Endpoints ────────────────────────────────────────────
@app.get("/chat/history")
async def chat_history(request: Request, limit: int = 10):
    identity = ensure_identity(request)
    rows = get_recent_history(identity.get("user_id"), identity.get("guest_id"), limit=limit)
    return {
        "items": rows,
        "is_authenticated": bool(identity.get("user_id")),
    }


@app.get("/chat/history/{entry_id}")
async def chat_history_item(entry_id: int, request: Request):
    identity = ensure_identity(request)
    item = get_history_item(entry_id, identity.get("user_id"), identity.get("guest_id"))
    if not item:
        raise HTTPException(status_code=404, detail="History item not found")
    return item


@app.get("/api/keywords")
def get_keywords():
    return {
        "total": len(ALL_KEYWORDS),
        "keywords": ALL_KEYWORDS[:100],
        "message": f"Total {len(ALL_KEYWORDS)} keywords available. Use /api/keywords/search?q=query for specific searches."
    }

@app.get("/api/keywords/search")
def search_keywords(q: str = ""):
    if not q:
        return {"keywords": ALL_KEYWORDS[:50]}
    query_lower = q.lower()
    matched = [kw for kw in ALL_KEYWORDS if query_lower in kw.lower()]
    return {"query": q, "total_matches": len(matched), "keywords": matched[:50]}

@app.get("/api/categories")
def get_categories():
    return {
        "total_categories": len(PRODUCT_CATEGORIES),
        "categories": {
            cat: {"name": cat, "count": len(items), "items": items[:10]}
            for cat, items in PRODUCT_CATEGORIES.items()
        }
    }

@app.get("/api/brands")
def get_brands():
    return {"total": len(SMART_HOME_BRANDS), "brands": sorted(SMART_HOME_BRANDS)}

@app.get("/api/features")
def get_features():
    return {"total": len(FEATURE_KEYWORDS), "features": FEATURE_KEYWORDS}

@app.get("/api/price-ranges")
def get_price_ranges():
    return {"ranges": PRICE_RANGES}

@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "version": "3.1.0",
        "mode": "ai_advisor",
        "services": {
            "groq_ai": "configured" if GROQ_API_KEY else "missing",
            "serpapi": "configured" if SERPAPI_KEY else "missing"
        },
        "features": {
            "total_keywords": len(ALL_KEYWORDS),
            "categories": len(PRODUCT_CATEGORIES),
            "brands": len(SMART_HOME_BRANDS)
        }
    }


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))




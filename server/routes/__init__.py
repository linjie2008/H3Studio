"""路由注册表：顺序与原 app.py 的 if 链完全一致。

first-match-wins：dispatch() 按 ORDER 顺序求值。
同一方法内如有匹配重叠（如 /pack 与 /director-pack），
顺序即语义；新增路由请追加到对应模块并登记在 ORDER 末尾。
"""
from __future__ import annotations

from . import system
from . import ai
from . import agent
from . import skills
from . import videos
from . import projects
from . import prompts
from . import assets
from . import media
from . import runs
from . import jobs
from . import packs
from . import downloader
from . import webapp

ALL = [
    system.ROUTES,
    ai.ROUTES,
    agent.ROUTES,
    skills.ROUTES,
    videos.ROUTES,
    projects.ROUTES,
    prompts.ROUTES,
    assets.ROUTES,
    media.ROUTES,
    runs.ROUTES,
    jobs.ROUTES,
    packs.ROUTES,
    downloader.ROUTES,
    webapp.ROUTES,
]

ORDER = [
    ('GET', 'get_comfy_ws'),
    ('GET', 'get_index'),
    ('GET', 'get_settings_hub_asset'),
    ('GET', 'get_api_version'),
    ('GET', 'get_api_ai_settings'),
    ('GET', 'get_api_ai_provider'),
    ('GET', 'get_api_product_agent_status'),
    ('GET', 'get_api_ai_minimax_settings'),
    ('GET', 'get_api_ai_mimo_settings'),
    ('GET', 'get_api_ai_models'),
    ('GET', 'get_api_ai_status'),
    ('GET', 'get_api_storage_status'),
    ('GET', 'get_api_skills'),
    ('GET', 'get_api_skills_preview'),
    ('GET', 'get_api_videos'),
    ('GET', 'get_api_health'),
    ('GET', 'get_api_models'),
    ('GET', 'get_api_projects'),
    ('GET', 'get_api_projects_director_pack'),
    ('GET', 'get_api_projects_pack'),
    ('GET', 'get_api_projects_shot_versions'),
    ('GET', 'get_api_project'),
    ('GET', 'get_api_assets'),
    ('GET', 'get_api_runs_report'),
    ('GET', 'get_api_run'),
    ('GET', 'get_api_runs'),
    ('GET', 'get_api_profile_images'),
    ('GET', 'get_api_profile_image'),
    ('GET', 'get_api_multiviews'),
    ('GET', 'get_api_multiview'),
    ('GET', 'get_media_assets'),
    ('GET', 'get_media_outputs'),
    ('POST', 'post_api_product_agent_chat'),
    ('POST', 'post_api_product_agent_confirm'),
    ('GET', 'get_api_creative_agent_status'),
    ('GET', 'get_api_creative_agent_runs'),
    ('POST', 'post_api_creative_agent_run'),
    ('POST', 'post_api_creative_agent_run_stream'),
    ('POST', 'post_api_creative_agent_confirm_stream'),
    ('POST', 'post_api_creative_agent_confirm'),
    ('POST', 'post_api_creative_agent_resume'),
    ('POST', 'post_api_creative_agent_undo'),
    ('POST', 'post_api_videos_import'),
    ('POST', 'post_api_projects_create'),
    ('POST', 'post_api_projects_clone'),
    ('POST', 'post_api_projects_batch_delete'),
    ('POST', 'post_api_skill_agent_parameters'),
    ('POST', 'post_api_skill_agent_recommend'),
    ('POST', 'post_api_translate_stream'),
    ('POST', 'post_api_translate'),
    ('POST', 'post_api_enhance_sequence'),
    ('POST', 'post_api_image_prompt'),
    ('POST', 'post_api_enhance'),
    ('POST', 'post_api_storyboard'),
    ('POST', 'post_api_skills_import'),
    ('POST', 'post_api_videos_delete'),
    ('POST', 'post_api_ai_test'),
    ('POST', 'post_api_ai_minimax_test'),
    ('POST', 'post_api_ai_mimo_test'),
    ('POST', 'post_api_ai_models'),
    ('POST', 'post_api_ai_settings'),
    ('POST', 'post_api_ai_provider'),
    ('POST', 'post_api_ai_minimax_settings'),
    ('POST', 'post_api_ai_mimo_settings'),
    ('POST', 'post_api_timeline_edit'),
    ('POST', 'post_api_prompt_preview'),
    ('POST', 'post_api_assets_upload'),
    ('POST', 'post_api_assets_extract_audio'),
    ('POST', 'post_api_assets_batch_delete'),
    ('POST', 'post_api_packs_import'),
    ('POST', 'post_api_projects_batch_multiview'),
    ('POST', 'post_api_projects_character_multiview'),
    ('POST', 'post_api_projects_entity_image'),
    ('POST', 'post_api_profile_image_cancel'),
    ('POST', 'post_api_multiview_cancel'),
    ('POST', 'post_api_projects_shots'),
    ('POST', 'post_api_projects_split'),
    ('POST', 'post_api_projects_remerge'),
    ('POST', 'post_api_projects_run'),
    ('POST', 'post_api_run_post_upscale'),
    ('POST', 'post_api_runs_decision'),
    ('POST', 'post_api_runs_resume'),
    ('DELETE', 'delete_api_ai_settings'),
    ('DELETE', 'delete_api_ai_minimax_settings'),
    ('DELETE', 'delete_api_ai_mimo_settings'),
    ('DELETE', 'delete_api_skills'),
    ('DELETE', 'delete_api_assets'),
    ('DELETE', 'delete_api_projects'),
    ('PUT', 'put_api_projects'),
    ('GET', 'get_app_index'),
    ('GET', 'get_app_asset'),
    ('POST', 'post_api_enhance_global'),
    ('GET', 'get_api_hardware'),
    ('POST', 'post_api_downloader_download'),
]


def build_registry():
    index = {(route.method, route.name): route
             for group in ALL for route in group}
    missing = [key for key in ORDER if key not in index]
    if missing:
        raise RuntimeError(f"路由缺失: {missing}")
    unused = [key for key in index if key not in set(ORDER)]
    if unused:
        raise RuntimeError(f"路由未注册: {unused}")
    return [index[key] for key in ORDER]


ROUTES = build_registry()

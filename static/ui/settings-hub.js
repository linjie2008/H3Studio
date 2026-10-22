/* --- Modern Dynamic Settings Hub (Contextual Titles, Zero-Nesting, Clean Workspaces) --- */
(function() {
  const projectSettingsPage = document.querySelector('[data-settings-page-content="project"]');
  if (!projectSettingsPage) return;

  const advPage = document.querySelector('[data-settings-page-content="advanced"]');
  const origGrid = projectSettingsPage.querySelector(':scope > .grid3');

  // Keep the stylesheet at the original insertion point for cascade order.
  const hubStyle = document.createElement('link');
  hubStyle.rel = 'stylesheet';
  hubStyle.href = '/ui/settings-hub.css';
  document.head.append(hubStyle);

  // Category definitions with dynamic contextual titles and descriptions
  const CATEGORIES = {
    canvas: {
      name: '画幅与工程',
      icon: '📐',
      title: '画幅与工程配置',
      desc: '设置项目名称、横竖屏、画布尺寸和全片导出方式，修改后影响整个项目。',
      hint: '项目名称、画幅、分辨率与导出'
    },
    skills: {
      name: '创作 Skill',
      icon: '🎨',
      title: '当前项目的创作 Skill',
      desc: '选择本项目的提示词 Skill 和参数建议。',
      hint: '仅影响当前项目'
    },
    inference: {
      name: 'H3 推理与显存',
      icon: '⚡',
      title: 'H3 视频生成推理与显存策略',
      desc: '选择 H3 的注意力、文本编码器、视频 VAE 和显存策略；这些设置影响生成速度与显存占用。',
      hint: '注意力、编码器与显存'
    },
    enhancement: {
      name: '超分与画面增强',
      icon: '🔍',
      title: 'SeedVR2 超分与画面接缝平滑',
      desc: '设置 SeedVR2 超分、人脸细节增强和分段视频的接缝处理。',
      hint: '超分、人脸与拼接接缝'
    },
    advanced: {
      name: '高级模型参数',
      icon: '🧪',
      title: '高级生成与精修参数',
      desc: '调整 SelfLift、二次采样、3D latent 放大和语义桥等进阶模型参数。',
      hint: 'SelfLift、二采与语义桥',
      badge: '实验'
    },
    global: {
      name: '全片共用内容',
      icon: '📝',
      title: '全片统一提示词与共享素材',
      desc: '填写全片共用的提示词，并选择会自动传给每个镜头的参考素材。',
      hint: '全片提示词与共享素材'
    },
    ai_services: {
      name: 'AI 服务与密钥',
      icon: '🤖',
      title: '本机 AI 服务',
      desc: '配置提示词模型和 API Key；设置对本机所有项目生效。',
      hint: '本机所有项目共用'
    },
    storage: {
      name: '数据库与环境',
      icon: '💾',
      title: '本地数据库与存储状态',
      desc: '查看本机项目数据库、媒体文件与生成记录的存储状态。',
      hint: '项目数据与存储状态'
    }
  };

  // Build Shell
  const hubShell = document.createElement('div');
  hubShell.className = 'settings-hub-shell';
  hubShell.innerHTML = `
    <header class="settings-hub-header">
      <div class="settings-hub-title-group">
        <div class="settings-hub-icon">⚙️</div>
        <div>
          <h2 style="font-size:17px;margin:0;font-weight:700">设置</h2>
          <p class="settings-hub-subtitle">
            <span class="settings-hub-badge" id="settingsHubScope">当前项目</span>
            <span id="settingsHubProjectName">当前项目</span>
            <span id="settingsHubCategoryBadge" style="margin-left:8px;color:#55626d;font-weight:600">› 画幅与工程</span>
          </p>
        </div>
      </div>
      <div class="settings-hub-actions">
        <div class="settings-search-box">
          <span class="settings-search-icon">🔍</span>
          <input type="search" id="settingsSearchInput" placeholder="查找任意设置项（如：SeedVR、SelfLift...）">
          <span id="settingsSearchCount" class="settings-search-badge hidden">0 匹配</span>
        </div>
        <div class="settings-view-switch">
          <button type="button" class="settings-view-btn active" data-settings-view="tabbed" title="单类别聚焦查看">📑 分类聚焦</button>
          <button type="button" class="settings-view-btn" data-settings-view="all" title="平铺展开全部设置">📜 全部展开</button>
        </div>
        <button type="button" class="settings-back-btn" data-workspace="create">✦ 返回创作</button>
      </div>
    </header>
    <div class="settings-hub-body">
      <nav class="settings-hub-sidebar" aria-label="设置类别">
        <div class="settings-nav-group">当前项目 <small>只影响当前项目</small></div>
        ${Object.entries(CATEGORIES).filter(([key]) => !['ai_services', 'storage'].includes(key)).map(([key, cat], idx) => `
          <button type="button" class="settings-nav-item ${idx === 0 ? 'active' : ''}" data-settings-cat="${key}">
            <span class="settings-nav-icon">${cat.icon}</span>
            <span class="settings-nav-copy"><span class="settings-nav-label">${cat.name}</span><small class="settings-nav-help">${cat.hint}</small></span>
            ${cat.badge ? `<span class="settings-nav-badge">${cat.badge}</span>` : ''}
          </button>
        `).join('')}
        <div class="settings-nav-group">本机全局 <small>对所有项目生效</small></div>
        ${Object.entries(CATEGORIES).filter(([key]) => ['ai_services', 'storage'].includes(key)).map(([key, cat]) => `
          <button type="button" class="settings-nav-item" data-settings-cat="${key}">
            <span class="settings-nav-icon">${cat.icon}</span>
            <span class="settings-nav-copy"><span class="settings-nav-label">${cat.name}</span><small class="settings-nav-help">${cat.hint}</small></span>
          </button>
        `).join('')}
      </nav>
      <main class="settings-hub-content" id="settingsHubContent"></main>
    </div>
  `;

  function makeCard(title, desc='', icon='', gridClass='grid3') {
    const card = document.createElement('div');
    card.className = 'settings-card';
    card.innerHTML = `
      <div class="settings-card-head">
        ${icon ? `<span class="settings-card-icon">${icon}</span>` : ''}
        <div>
          <h4>${title}</h4>
          ${desc ? `<p class="hint">${desc}</p>` : ''}
        </div>
      </div>
      <div class="settings-card-body ${gridClass}"></div>
    `;
    return card;
  }

  function extractField(selector, root=origGrid) {
    if (!root) return null;
    const el = root.querySelector(selector);
    return el ? (el.closest('.field') || el) : null;
  }

  const contentHost = hubShell.querySelector('#settingsHubContent');

  function makePanel(id) {
    const p = document.createElement('div');
    p.className = 'settings-cat-panel';
    p.dataset.settingsPanel = id;
    const intro = document.createElement('div');
    intro.className = 'settings-panel-intro';
    intro.innerHTML = `<h3>${CATEGORIES[id].title}</h3><p>${CATEGORIES[id].desc}</p>`;
    p.append(intro);
    return p;
  }

  // 1. Canvas
  const pCanvas = makePanel('canvas');
  const cardCanvas = makeCard('画布尺寸与画幅预设', '设置横屏 16:9 / 竖屏 9:16 及像素档位，项目画布会同步应用到所有镜头。', '📐', 'grid3');
  for (const s of ['[data-canvas-orientation]', '[data-canvas-preset="project"]', '[data-project="width"]', '[data-project="height"]']) {
    const f = extractField(s);
    if (f) cardCanvas.querySelector('.settings-card-body').append(f);
  }
  const cardProject = makeCard('工程设置与全片导出', '项目基础命名与多镜头导出方式。', '🎬', 'grid3');
  for (const s of ['[data-project="name"]', '[data-project="continuous_render_mode"]', '[data-project="export_mode"]']) {
    const f = extractField(s);
    if (f) cardProject.querySelector('.settings-card-body').append(f);
  }
  pCanvas.append(cardCanvas, cardProject);

  // 2. Project Skill and machine-wide AI credentials have different scopes.
  const pAiServices = makePanel('ai_services');
  const cardAiKeys = makeCard('AI 提示词模型与云端 API', '配置 DeepSeek 与小米 MiMo 密钥；用量按对应平台账户计费。', '🤖', 'settings-card-body-stack');
  const promptProviderEl = projectSettingsPage.querySelector('section.character-library:has(#promptProvider)');
  if (promptProviderEl) cardAiKeys.querySelector('.settings-card-body').append(promptProviderEl);
  const deepseekDetails = projectSettingsPage.querySelector('details.character-library:has(#deepseekKey)');
  if (deepseekDetails) {
    deepseekDetails.open = false;
    cardAiKeys.querySelector('.settings-card-body').append(deepseekDetails);
  }
  const mimoDetails = projectSettingsPage.querySelector('details.character-library:has(#mimoKey)');
  if (mimoDetails) {
    mimoDetails.open = false;
    cardAiKeys.querySelector('.settings-card-body').append(mimoDetails);
  }

  pAiServices.append(cardAiKeys);

  const pSkills = makePanel('skills');
  const cardSkills = makeCard('项目级提示词 Skill', '绑定到当前项目：全片镜头的 AI 提示词优化和 AI 分镜均使用此 Skill。', '🎨', 'settings-card-body-stack');
  const skillDetails = projectSettingsPage.querySelector('details.character-library:has(#promptSkillSelect)');
  if (skillDetails) {
    skillDetails.open = true;
    cardSkills.querySelector('.settings-card-body').append(skillDetails);
  }
  const skillAgentPanel = projectSettingsPage.querySelector('section.character-library.skill-agent-panel');
  if (skillAgentPanel) cardSkills.querySelector('.settings-card-body').append(skillAgentPanel);

  pSkills.append(cardSkills);

  // 3. Inference & VRAM
  const pInference = makePanel('inference');
  const cardInference = makeCard('H3 注意力与编码模型', '选择注意力加速算法、文本编码器与视频 VAE。', '⚡', 'grid3');
  for (const s of ['[data-project="attention_mode"]', '#h3ClipModel', '#h3VideoVae']) {
    const f = extractField(s);
    if (f) cardInference.querySelector('.settings-card-body').append(f);
  }
  const cardVram = makeCard('显存策略与镜头接续', '自动显存调节及长片段间接续帧引导。', '💾', 'grid3');
  for (const s of ['[data-project="memory_policy"]', '[data-project="continuity_frames"]', '[data-project="identity_first_on_switch"]']) {
    const f = extractField(s);
    if (f) cardVram.querySelector('.settings-card-body').append(f);
  }
  pInference.append(cardInference, cardVram);

  // 4. Enhancement
  const pEnhance = makePanel('enhancement');
  const cardSeedVR = makeCard('SeedVR2 视频超分', '生成 H3 原片后进行高分辨率重建。', '🔍', 'grid3');
  for (const s of ['[data-project="upscale"]', '[data-project="upscale_resolution"]', '[data-project="confirm_upscale"]']) {
    const f = extractField(s);
    if (f) cardSeedVR.querySelector('.settings-card-body').append(f);
  }
  const cardDetails = makeCard('人脸细节与分段视频拼接', '人脸检测增强与分段拼接接缝亮度匹配。', '👤', 'settings-card-body-stack');
  const faceField = extractField('[data-project="face_refine"]');
  if (faceField) cardDetails.querySelector('.settings-card-body').append(faceField);
  const seamEl = projectSettingsPage.querySelector('#seamSettings');
  if (seamEl) {
    seamEl.open = true;
    cardDetails.querySelector('.settings-card-body').append(seamEl);
  }
  pEnhance.append(cardSeedVR, cardDetails);

  // 5. Advanced Model Ops (from advPage)
  const pAdv = makePanel('advanced');
  const cardSelfLift = makeCard('SelfLift 渐进一采', '低清前缀快速出图后转入 H3 3D latent 高清收尾。', '🚀', 'grid3');
  if (advPage) {
    for (const s of ['[data-project="selflift"]', '[data-project="selflift_scale"]', '[data-project="selflift_highres_steps"]', '[data-project="selflift_rho"]', '[data-project="selflift_tiling"]', '[data-project="selflift_tile_count"]', '[data-project="selflift_tile_overlap"]']) {
      const f = extractField(s, advPage);
      if (f) cardSelfLift.querySelector('.settings-card-body').append(f);
    }
  }
  const cardRefine = makeCard('H3 3D Latent 权重与二采精修', '原画布或 3D latent 放大后的多轮重绘精修。', '💎', 'grid3');
  if (advPage) {
    for (const s of ['#latentModel', '[data-project="refine_mode"]', '[data-project="refine_steps"]', '[data-project="refine_passes"]', '[data-project="refine_denoise"]', '[data-project="refine_width"]']) {
      const f = extractField(s, advPage);
      if (f) cardRefine.querySelector('.settings-card-body').append(f);
    }
  }
  const cardBridge = makeCard('语义桥适配器与显存调度', '适配器增强及镜头间显存释放策略。', '🌉', 'grid3');
  if (advPage) {
    for (const s of ['#semanticAdapter', '[data-project="semantic_alpha"]', '[data-project="semantic_magnitude_match"]', '[data-project="clear_vram_between_shots"]']) {
      const f = extractField(s, advPage);
      if (f) cardBridge.querySelector('.settings-card-body').append(f);
    }
  }
  pAdv.append(cardSelfLift, cardRefine, cardBridge);

  // 6. Global Requirements
  const pGlobal = makePanel('global');
  const cardGlobalPrompt = makeCard('全片统一提示词', '填写全片固定视觉与运镜要求，会自动附加给所有镜头。', '📜', 'settings-card-body-stack');
  const sharedPromptEl = projectSettingsPage.querySelector('[data-project="shared_prompt"]');
  if (sharedPromptEl) cardGlobalPrompt.querySelector('.settings-card-body').append(sharedPromptEl);
  const cardSharedRefs = makeCard('全片共享参考素材', '选中的参考图/视频/音频会自动关联到项目中所有镜头。', '🖼️', 'settings-card-body-stack');
  const sharedRefsEl = projectSettingsPage.querySelector('#sharedRefs');
  if (sharedRefsEl) cardSharedRefs.querySelector('.settings-card-body').append(sharedRefsEl);
  pGlobal.append(cardGlobalPrompt, cardSharedRefs);

  // 7. Storage
  const pStorage = makePanel('storage');
  const cardStorage = makeCard('项目数据库与系统存储', '项目配置与生成记录存储在本地 SQLite 数据库中。', '💾', 'settings-card-body-stack');
  const storageDetails = projectSettingsPage.querySelector('details.character-library:has(#storageStatus)');
  if (storageDetails) {
    storageDetails.open = true;
    cardStorage.querySelector('.settings-card-body').append(storageDetails);
  }
  pStorage.append(cardStorage);

  contentHost.append(pCanvas, pSkills, pInference, pEnhance, pAdv, pGlobal, pAiServices, pStorage);

  // Replace projectSettingsPage contents with hubShell
  projectSettingsPage.replaceChildren(hubShell);

  let activeCat = 'canvas';
  let viewMode = 'tabbed';

  window.switchSettingsCategory = function(cat) {
    if (!CATEGORIES[cat]) cat = 'canvas';
    activeCat = cat;

    // Update dynamic header breadcrumb
    const meta = CATEGORIES[cat];
    const catBadge = hubShell.querySelector('#settingsHubCategoryBadge');
    if (catBadge) catBadge.textContent = `› ${meta.name}（${meta.hint}）`;
    const machineWide = ['ai_services', 'storage'].includes(cat);
    hubShell.querySelector('#settingsHubScope').textContent = machineWide ? '本机全局' : '当前项目';
    hubShell.querySelector('#settingsHubProjectName').classList.toggle('hidden', machineWide);

    hubShell.querySelectorAll('.settings-nav-item').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.settingsCat === cat);
    });
    hubShell.querySelectorAll('.settings-cat-panel').forEach(panel => {
      if (viewMode === 'tabbed') {
        panel.classList.toggle('hidden', panel.dataset.settingsPanel !== cat);
      } else {
        panel.classList.remove('hidden');
      }
    });
    if (viewMode === 'all') {
      const target = hubShell.querySelector(`[data-settings-panel="${cat}"]`);
      target?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  hubShell.querySelector('.settings-hub-sidebar').addEventListener('click', e => {
    const btn = e.target.closest('[data-settings-cat]');
    if (btn) switchSettingsCategory(btn.dataset.settingsCat);
  });

  hubShell.querySelector('.settings-view-switch').addEventListener('click', e => {
    const btn = e.target.closest('[data-settings-view]');
    if (!btn) return;
    viewMode = btn.dataset.settingsView;
    hubShell.querySelectorAll('[data-settings-view]').forEach(b => b.classList.toggle('active', b === btn));
    switchSettingsCategory(activeCat);
  });

  const searchInput = hubShell.querySelector('#settingsSearchInput');
  const searchBadge = hubShell.querySelector('#settingsSearchCount');
  searchInput.addEventListener('input', e => {
    const q = e.target.value.trim().toLowerCase();
    if (!q) {
      searchBadge.classList.add('hidden');
      hubShell.querySelectorAll('.settings-card').forEach(c => c.classList.remove('search-hidden'));
      switchSettingsCategory(activeCat);
      return;
    }
    hubShell.querySelectorAll('.settings-cat-panel').forEach(p => p.classList.remove('hidden'));
    let matches = 0;
    hubShell.querySelectorAll('.settings-card').forEach(card => {
      const matched = card.textContent.toLowerCase().includes(q);
      card.classList.toggle('search-hidden', !matched);
      if (matched) matches++;
    });
    searchBadge.textContent = `${matches} 处设置`;
    searchBadge.classList.remove('hidden');
  });

  switchSettingsCategory('canvas');

  // Synchronize with activeWorkspace & render()
  const origRender = window.render;
  window.render = function() {
    if (origRender) origRender();
    const nameEl = document.getElementById('settingsHubProjectName');
    if (nameEl && project) nameEl.textContent = project.name || '当前项目';
    if (['project', 'advanced'].includes(activeWorkspace)) {
      document.querySelectorAll('[data-settings-page-content]').forEach(el => {
        el.classList.toggle('hidden', el.dataset.settingsPageContent !== 'project');
      });
      if (activeWorkspace === 'advanced') {
        switchSettingsCategory('advanced');
      } else {
        switchSettingsCategory(activeCat);
      }
    }
  };
})();

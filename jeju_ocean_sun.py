# -*- coding: utf-8 -*-
"""
제주도 DEM + 물결치는 바다 + 하늘을 가로지르는 태양  (Blender 4.x / 5.x, Cycles)

사용법
  1) Blender → Scripting 탭 → Open 으로 이 파일을 열고 ▶ Run Script
     또는 터미널:  blender --python jeju_ocean_sun.py
  2) 렌더:  F12 (현재 프레임)  /  Ctrl+F12 (애니메이션)
  3) 실제 DEM 을 쓰려면 아래 DEM_PATH 에 파일 경로를 적으세요. (비워 두면 가상의 제주도/한라산 지형을 만듭니다)

구성
  - 지형 : DEM(.asc / .npy / .tif / .png 높이맵) → 메시.  없으면 한라산(1947m) + 오름 360개 느낌의 가상 지형
  - 바다 : Ocean 모디파이어(실제 물결 변형, 시간 애니메이션) + 투명한 물 셰이더 + 잔물결 노멀(태양 반짝임)
  - 하늘 : Sky Texture(대기 산란, 태양 원반 포함).  제주도 위도/날짜 기준 천문 공식으로 태양이 동→남→서로 이동
  - 태양 빛이 바다 표면에 반사되어 반짝이는 길(sun glitter)이 Cycles 경로 추적으로 자연스럽게 생깁니다.

좌표: +X = 동쪽, +Y = 북쪽, +Z = 위.  1 Blender 단위(BU) = 100 m (UNIT_M).
"""
import bpy
import math
import os
import sys
import numpy as np
from mathutils import Vector

# =============================================================================
# 설정 (여기만 고치면 됩니다)
# =============================================================================
DEM_PATH = ""                 # 예: r"C:\data\jeju_dem.asc"  (.asc .npy .tif .png .jpg). 빈 문자열 = 가상 지형
DEM_EXTENT_KM = (96.0, 64.0)  # DEM 이 덮는 가로(동서) x 세로(남북) 실제 크기(km). 가상 지형일 때도 사용
DEM_MAX_ELEV_M = 1947.0       # 이미지(0~1 정규화) 높이맵일 때 흰색 = 몇 m 인지 (한라산 1947m)
DEM_NORTH_UP = True           # DEM 위쪽이 북쪽이면 True (대부분 True)

VERT_EXAGGERATION = 2.5       # 지형 높이 과장 배율 (1.0 = 실제 비율)
SEA_DEPTH_M = 70.0            # 먼 바다의 수심(m) - DEM 에 수심이 없을 때 만들어 줌
GRID_COLUMNS = 480            # 지형 메시 가로 정점 수 (클수록 세밀 / 느림)

# 태양 (제주시 부근)
LATITUDE = 33.4               # 위도
SUN_MONTH_DAY = (12, 21)      # 날짜(월, 일). 동지: 태양이 낮게 떠 화면(하늘)에 계속 보임 / 추분(9,23), 하지(6,21)는 더 높이 뜸
SUN_START_HOUR = 7.5          # 애니메이션 시작 시각 (태양시, 시)
SUN_END_HOUR = 16.5          # 애니메이션 끝 시각
SUN_SIZE_DEG = 0.8            # 태양 원반 크기(실제 0.53°, 보기 좋게 키움)
SUN_INTENSITY = 1.0

# 바다
TILE_BU = 100.0               # 파도 타일 1장의 크기 (BU, 100 BU = 10 km)
TILE_REPEAT = 15              # 타일 반복 수 (15 x 100 BU = 150 km 바다)
OCEAN_RES = 10                # 파도 해상도 (렌더). 클수록 정밀하지만 무거움
WAVE_HEIGHT_SCALE = 1.4       # 파도 높이 배율 (Z). 크면 파도가 더 높아짐
WIND = 38.0                    # 풍속(클수록 큰 파도)
CHOPPINESS = 2.4              # 파도 마루의 뾰족함
FOAM_COVERAGE = -3.0          # 흰 물거품 양: 값이 클수록 많아짐 (-5 거의 없음 ~ 0 많음, 실측)
FOAM_THRESHOLD = 0.30         # 클수록 포말이 줄어듦 (0~0.9)
WAVE_TIME_PER_SEC = 0.7       # 1초 영상당 진행되는 파도 시간
REAL_WATER = True             # True = 투명한 물(바닥이 비침, 느림) / False = 불투명 물(빠름)

# 애니메이션 / 렌더
FPS = 24
FRAME_START = 1
FRAME_END = 360               # 24fps 기준 15초
RES_X, RES_Y = 1280, 720
SAMPLES = 64
USE_GPU = True                # GPU 가 있으면 사용

# 카메라 (BU)
CAMERA_POS = (40.0, 400.0, 22.0)   # 섬 북쪽 바다 위에서 남쪽(한라산 + 태양)을 바라봄
CAMERA_TARGET = (0.0, -40.0, 60.0)
CAMERA_LENS_MM = 16.0
EXPOSURE = -1.4                    # 화면 밝기 (낮추면 어두워짐)

UNIT_M = 100.0                # 1 BU = 100 m


# =============================================================================
# 명령줄 옵션 (테스트용):  blender -b -P jeju_ocean_sun.py -- --still 150 --out a.png --res 640x360 --samples 16
# =============================================================================
def parse_args():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    opts = {}
    i = 0
    while i < len(argv):
        if argv[i].startswith('--'):
            opts[argv[i][2:]] = argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith('--') else True
        i += 1
    return opts


# =============================================================================
# 유틸
# =============================================================================
def gauss_blur(a, sigma):
    """FFT 가우시안 블러 (scipy 없이)"""
    ny, nx = a.shape
    fy = np.fft.fftfreq(ny)[:, None]
    fx = np.fft.rfftfreq(nx)[None, :]
    k = np.exp(-2 * (math.pi ** 2) * sigma ** 2 * (fx ** 2 + fy ** 2))
    return np.fft.irfft2(np.fft.rfft2(a) * k, s=a.shape)


def resize(a, ny, nx):
    """쌍선형 리샘플링"""
    h, w = a.shape
    ys = np.linspace(0, h - 1, ny)
    xs = np.linspace(0, w - 1, nx)
    y0 = np.floor(ys).astype(int); y1 = np.minimum(y0 + 1, h - 1); fy = (ys - y0)[:, None]
    x0 = np.floor(xs).astype(int); x1 = np.minimum(x0 + 1, w - 1); fx = (xs - x0)[None, :]
    top = a[y0][:, x0] * (1 - fx) + a[y0][:, x1] * fx
    bot = a[y1][:, x0] * (1 - fx) + a[y1][:, x1] * fx
    return top * (1 - fy) + bot * fy


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def make_linear(id_data):
    """id_data 의 모든 키프레임을 선형 보간으로 (Blender 4.x / 5.x 액션 구조 모두 대응)"""
    ad = getattr(id_data, 'animation_data', None)
    if not ad or not ad.action:
        return
    act = ad.action
    curves = []
    if hasattr(act, 'fcurves'):
        curves = list(act.fcurves)
    else:                                    # Blender 5.x: 레이어드 액션
        slot = ad.action_slot
        for layer in act.layers:
            for strip in layer.strips:
                try:
                    curves += list(strip.channelbag(slot).fcurves)
                except Exception:
                    pass
    for fc in curves:
        for kp in fc.keyframe_points:
            kp.interpolation = 'LINEAR'


def set_input(node, names, value):
    """Blender 버전마다 달라지는 소켓 이름 대응"""
    if isinstance(names, str):
        names = [names]
    for n in names:
        if n in node.inputs:
            node.inputs[n].default_value = value
            return True
    return False


# =============================================================================
# DEM
# =============================================================================
def load_dem_file(path):
    """반환: 고도(m) 2차원 배열, 행 0 = 남쪽(j 증가 = 북쪽)"""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        a = np.load(path).astype(np.float64)
        a = a[::-1] if DEM_NORTH_UP else a
    elif ext in ('.asc', '.txt'):
        header, rows = {}, []
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                p = line.split()
                if len(p) == 2 and p[0].lower() in ('ncols', 'nrows', 'xllcorner', 'yllcorner', 'xllcenter', 'yllcenter', 'cellsize', 'nodata_value'):
                    header[p[0].lower()] = float(p[1])
                elif p:
                    rows.append(np.array(p, dtype=np.float64))
        a = np.vstack(rows)
        nod = header.get('nodata_value', -9999.0)
        a[a == nod] = 0.0
        a = a[::-1]
    else:  # 이미지(GeoTIFF/PNG/EXR...)를 Blender 로 읽기
        img = bpy.data.images.load(path, check_existing=False)
        try:
            img.colorspace_settings.name = 'Non-Color'
        except Exception:
            pass
        w, h = img.size
        px = np.empty(w * h * 4, dtype=np.float32)
        img.pixels.foreach_get(px)
        a = px.reshape(h, w, 4)[:, :, 0].astype(np.float64)  # 행 0 = 이미지 맨 아래
        bpy.data.images.remove(img)
        if a.max() <= 1.5:
            a = a * DEM_MAX_ELEV_M          # 0~1 정규화 이미지
        if not DEM_NORTH_UP:
            a = a[::-1]
    a[a < -500] = 0.0
    return a


def procedural_island(ny, nx, extent_km, seed=7):
    """제주도 느낌의 가상 지형: 타원형 섬 + 한라산(1947m) + 크레이터 + 오름 360개"""
    rng = np.random.default_rng(seed)
    wk, hk = extent_km
    x = np.linspace(-wk / 2, wk / 2, nx)[None, :] * np.ones((ny, 1))
    y = np.linspace(-hk / 2, hk / 2, ny)[:, None] * np.ones((1, nx))

    ang = math.radians(6.0)                        # 섬의 장축을 살짝 기울임
    xr = x * math.cos(ang) + y * math.sin(ang)
    yr = -x * math.sin(ang) + y * math.cos(ang)
    a_km, b_km = 36.5, 15.8                        # 동서 73km, 남북 약 31km

    noise = gauss_blur(rng.standard_normal((ny, nx)), 18)
    noise /= max(1e-6, np.abs(noise).max())
    r = np.sqrt((xr / a_km) ** 2 + (yr / b_km) ** 2) * (1.0 + 0.10 * noise)
    s = np.clip(1.0 - r, 0.0, 1.0)

    # 한라산: 섬 중앙 조금 서쪽
    hx, hy = -3.0, 0.0
    rh = np.sqrt(((x - hx) / 1.0) ** 2 + ((y - hy) / 1.0) ** 2)
    h = 1950.0 * (s ** 2.4)
    h *= 1.0 + 0.55 * np.exp(-(rh / 9.0) ** 2)     # 중앙을 더 봉긋하게
    h = np.minimum(h, 1947.0)
    h -= 130.0 * np.exp(-(rh / 0.55) ** 2) * (s > 0.5)   # 백록담 분화구

    # 오름 360개
    land_mask = s > 0.12
    ys_i, xs_i = np.nonzero(land_mask)
    pick = rng.choice(len(ys_i), size=360)
    for k in pick:
        px_, py_ = x[ys_i[k], xs_i[k]], y[ys_i[k], xs_i[k]]
        rad = rng.uniform(0.5, 1.3)
        hh = rng.uniform(60, 280) * (0.4 + s[ys_i[k], xs_i[k]])
        h += hh * np.exp(-(((x - px_) ** 2 + (y - py_) ** 2) / rad ** 2)) * smoothstep(s * 6)

    h *= 1.0 + 0.10 * gauss_blur(rng.standard_normal((ny, nx)), 2.0) * 4
    # 잔 지형(계곡·능선) 디테일
    for sig, amp in ((6, 70.0), (2.5, 22.0)):
        d = gauss_blur(rng.standard_normal((ny, nx)), sig)
        h += amp * d / max(1e-6, np.abs(d).max()) * smoothstep(s * 5)
    h = np.where(s > 0.0, np.maximum(h, 0.0), 0.0)
    return h


def make_heightfield():
    """고도(m) 배열을 [ny, nx] 로 반환 (행 0 = 남쪽). 바다 수심 포함"""
    wk, hk = DEM_EXTENT_KM
    nx = GRID_COLUMNS
    ny = int(round(nx * hk / wk))
    if DEM_PATH and os.path.exists(DEM_PATH):
        print('[Jeju] DEM 불러오는 중:', DEM_PATH)
        h = resize(load_dem_file(DEM_PATH), ny, nx)
    else:
        if DEM_PATH:
            print('[Jeju] DEM 파일을 찾을 수 없어 가상 지형을 만듭니다:', DEM_PATH)
        else:
            print('[Jeju] DEM_PATH 가 비어 있어 가상 지형을 만듭니다.')
        h = procedural_island(ny, nx, DEM_EXTENT_KM)

    if h.min() >= -1.0:  # 수심 데이터가 없으면 해안에서 멀수록 깊어지게 만듦
        land = (h > 0.5).astype(np.float64)
        km_per_cell = wk / nx
        p = gauss_blur(land, sigma=max(2.0, 7.0 / km_per_cell))   # 해안 근접도
        depth = SEA_DEPTH_M * smoothstep((0.5 - p) / 0.5) ** 0.8
        h = np.where(h > 0.5, h, -depth)
    return h, nx, ny


# =============================================================================
# 씬 만들기
# =============================================================================
def clear_scene():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.cameras, bpy.data.lights, bpy.data.worlds):
        for d in list(coll):
            if d.users == 0:
                coll.remove(d)


def build_terrain(h, nx, ny):
    wk, hk = DEM_EXTENT_KM
    w_bu, h_bu = wk * 1000.0 / UNIT_M, hk * 1000.0 / UNIT_M
    xs = np.linspace(-w_bu / 2, w_bu / 2, nx)
    ys = np.linspace(-h_bu / 2, h_bu / 2, ny)
    X, Y = np.meshgrid(xs, ys)
    Z = h * VERT_EXAGGERATION / UNIT_M
    verts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    idx = np.arange(nx * ny).reshape(ny, nx)
    quads = np.stack([idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()], axis=1)

    mesh = bpy.data.meshes.new('JejuTerrain')
    mesh.from_pydata(verts.tolist(), [], quads.tolist())
    mesh.update()
    obj = bpy.data.objects.new('JejuTerrain', mesh)
    bpy.context.scene.collection.objects.link(obj)
    for p in mesh.polygons:
        p.use_smooth = True

    # 바다 밑바닥(지형 바깥쪽)
    floor_z = float(Z.min())
    fm = bpy.data.meshes.new('SeaFloor')
    big = 4000.0
    fm.from_pydata([(-big, -big, floor_z), (big, -big, floor_z), (big, big, floor_z), (-big, big, floor_z)], [], [(0, 1, 2, 3)])
    fo = bpy.data.objects.new('SeaFloor', fm)
    bpy.context.scene.collection.objects.link(fo)
    fo.location.z = -0.002   # 지형 가장자리와 겹침 방지

    return obj, fo


def make_terrain_material():
    mat = bpy.data.materials.new('TerrainMat')
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
    geo = nt.nodes.new('ShaderNodeNewGeometry')
    sep = nt.nodes.new('ShaderNodeSeparateXYZ')
    mp = nt.nodes.new('ShaderNodeMapRange')
    ramp = nt.nodes.new('ShaderNodeValToRGB')
    noise = nt.nodes.new('ShaderNodeTexNoise')
    mix = nt.nodes.new('ShaderNodeMix')
    mix.data_type = 'RGBA'; mix.blend_type = 'MULTIPLY'
    mix.inputs[0].default_value = 0.45

    # 높이(m) -> 0~1.  -70m ~ 1950m
    mp.inputs['From Min'].default_value = -SEA_DEPTH_M * VERT_EXAGGERATION / UNIT_M
    mp.inputs['From Max'].default_value = 1950.0 * VERT_EXAGGERATION / UNIT_M
    cr = ramp.color_ramp
    stops = [
        (0.000, (0.18, 0.16, 0.12, 1)),   # 깊은 바닥
        (0.030, (0.55, 0.50, 0.36, 1)),   # 해저 모래
        (0.036, (0.78, 0.72, 0.55, 1)),   # 해변 모래
        (0.050, (0.20, 0.38, 0.12, 1)),   # 해안 초지
        (0.250, (0.16, 0.32, 0.10, 1)),   # 중산간 숲
        (0.600, (0.30, 0.27, 0.16, 1)),   # 관목/억새
        (0.850, (0.30, 0.26, 0.24, 1)),   # 암석
        (1.000, (0.45, 0.43, 0.42, 1)),   # 정상부
    ]
    cr.elements[0].position, cr.elements[0].color = stops[0][0], stops[0][1]
    cr.elements[1].position, cr.elements[1].color = stops[1][0], stops[1][1]
    for pos, col in stops[2:]:
        e = cr.elements.new(pos)
        e.color = col
    noise.inputs['Scale'].default_value = 3.0
    noise.inputs['Detail'].default_value = 8.0

    nt.links.new(geo.outputs['Position'], sep.inputs['Vector'])
    nt.links.new(sep.outputs['Z'], mp.inputs['Value'])
    nt.links.new(mp.outputs['Result'], ramp.inputs['Fac'])
    nt.links.new(ramp.outputs['Color'], mix.inputs[6])
    nt.links.new(noise.outputs['Fac'], mix.inputs[7])
    nt.links.new(mix.outputs[2], bsdf.inputs['Base Color'])
    nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    set_input(bsdf, 'Roughness', 0.92)
    return mat


def make_water_material(tilt_noise_scale=4.0):
    mat = bpy.data.materials.new('OceanMat')
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
    texco = nt.nodes.new('ShaderNodeTexCoord')
    noise = nt.nodes.new('ShaderNodeTexNoise')
    noise.noise_dimensions = '4D'
    noise.inputs['Scale'].default_value = tilt_noise_scale
    noise.inputs['Detail'].default_value = 5.0
    noise.inputs['Roughness'].default_value = 0.6
    bump = nt.nodes.new('ShaderNodeBump')
    bump.inputs['Strength'].default_value = 1.0      # 잔물결 세기 (태양 반짝임이 퍼지는 정도)
    bump.inputs['Distance'].default_value = 0.1

    nt.links.new(texco.outputs['Object'], noise.inputs['Vector'])
    nt.links.new(noise.outputs['Fac'], bump.inputs['Height'])
    nt.links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])

    set_input(bsdf, 'IOR', 1.333)
    set_input(bsdf, 'Roughness', 0.025)
    if REAL_WATER:
        set_input(bsdf, 'Base Color', (0.85, 0.97, 1.0, 1))
        if not set_input(bsdf, ['Transmission Weight', 'Transmission'], 1.0):
            pass
        absorb = nt.nodes.new('ShaderNodeVolumeAbsorption')
        absorb.inputs['Color'].default_value = (0.12, 0.55, 0.72, 1)   # 붉은 빛이 먼저 흡수 → 청록색
        absorb.inputs['Density'].default_value = 16.0
        nt.links.new(absorb.outputs['Volume'], out.inputs['Volume'])
    else:
        set_input(bsdf, 'Base Color', (0.02, 0.18, 0.28, 1))
    # 파도 마루의 흰 물거품(포말): Ocean 모디파이어가 만든 'foam' 속성을 사용
    foam_attr = nt.nodes.new('ShaderNodeAttribute')
    foam_attr.attribute_type = 'GEOMETRY'
    foam_attr.attribute_name = 'foam'
    foam_ramp = nt.nodes.new('ShaderNodeValToRGB')
    foam_ramp.color_ramp.elements[0].position = FOAM_THRESHOLD
    foam_ramp.color_ramp.elements[1].position = FOAM_THRESHOLD + 0.35
    foam_bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
    set_input(foam_bsdf, 'Base Color', (0.95, 0.97, 1.0, 1))
    set_input(foam_bsdf, 'Roughness', 0.55)
    mix_sh = nt.nodes.new('ShaderNodeMixShader')
    nt.links.new(foam_attr.outputs['Fac'], foam_ramp.inputs['Fac'])
    nt.links.new(foam_ramp.outputs['Color'], mix_sh.inputs['Fac'])
    nt.links.new(bsdf.outputs['BSDF'], mix_sh.inputs[1])
    nt.links.new(foam_bsdf.outputs['BSDF'], mix_sh.inputs[2])
    nt.links.new(mix_sh.outputs['Shader'], out.inputs['Surface'])
    return mat, noise


def build_ocean(frame_start, frame_end):
    scene = bpy.context.scene
    bpy.ops.mesh.primitive_plane_add(size=1)
    ob = bpy.context.active_object
    ob.name = 'Ocean'
    m = ob.modifiers.new('Ocean', 'OCEAN')
    m.geometry_mode = 'GENERATE'
    m.spatial_size = 200                     # 파도의 물리적 크기(m): 파도 모양을 결정
    m.size = TILE_BU / 200.0                 # 타일 1장의 BU 폭 = spatial_size * size
    m.resolution = OCEAN_RES
    m.viewport_resolution = max(3, OCEAN_RES // 2)
    m.repeat_x = TILE_REPEAT
    m.repeat_y = TILE_REPEAT
    m.wind_velocity = WIND
    m.choppiness = CHOPPINESS
    m.wave_scale = 1.0
    m.wave_scale_min = 0.05
    m.damping = 0.5
    m.random_seed = 3
    m.use_normals = True
    m.use_foam = True
    m.foam_coverage = FOAM_COVERAGE
    try:
        m.foam_layer_name = 'foam'
    except Exception:
        pass
    # 타일 그리드가 [-w/2, (rep-0.5)w] 로 만들어지므로 중앙 정렬
    ob.location = (-(TILE_REPEAT - 1) * TILE_BU / 2.0, -(TILE_REPEAT - 1) * TILE_BU / 2.0, 0.0)
    ob.scale = (1.0, 1.0, WAVE_HEIGHT_SCALE)

    # 파도 시간 애니메이션 (선형)
    duration = (frame_end - frame_start) / float(FPS)
    m.time = 0.0
    m.keyframe_insert('time', frame=frame_start)
    m.time = duration * WAVE_TIME_PER_SEC
    m.keyframe_insert('time', frame=frame_end)

    mat, noise = make_water_material()
    ob.data.materials.append(mat)
    for p in ob.data.polygons:
        p.use_smooth = True

    # 수평선까지 이어지는 바깥쪽 평평한 바다(파도 타일 바깥을 감싸는 고리 모양)
    inner = TILE_REPEAT * TILE_BU / 2.0
    outer = 6000.0
    rv = [(-outer, -outer, 0), (outer, -outer, 0), (outer, outer, 0), (-outer, outer, 0),
          (-inner, -inner, 0), (inner, -inner, 0), (inner, inner, 0), (-inner, inner, 0)]
    rf = [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    rm = bpy.data.meshes.new('OceanFar')
    rm.from_pydata(rv, [], rf)
    rm.update()
    far = bpy.data.objects.new('OceanFar', rm)
    scene.collection.objects.link(far)
    far.data.materials.append(mat)

    # 잔물결(반짝임) 노이즈도 시간에 따라 흐르게
    noise.inputs['W'].default_value = 0.0
    noise.inputs['W'].keyframe_insert('default_value', frame=frame_start)
    noise.inputs['W'].default_value = duration * 0.9
    noise.inputs['W'].keyframe_insert('default_value', frame=frame_end)
    make_linear(ob)           # 파도 시간: 일정한 속도
    make_linear(mat.node_tree)
    return ob


def sun_state(hour):
    """태양시 hour 의 (고도°, 방위각°)  방위각: 북=0, 동=90, 남=180, 서=270"""
    month, day = SUN_MONTH_DAY
    cum = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    n = cum[month - 1] + day
    decl = math.radians(23.44) * math.sin(2 * math.pi * (284 + n) / 365.0)
    phi = math.radians(LATITUDE)
    H = math.radians((hour - 12.0) * 15.0)
    sin_el = math.sin(phi) * math.sin(decl) + math.cos(phi) * math.cos(decl) * math.cos(H)
    el = math.asin(max(-1.0, min(1.0, sin_el)))
    az = math.atan2(math.sin(H), math.cos(H) * math.sin(phi) - math.tan(decl) * math.cos(phi)) + math.pi
    return math.degrees(el), math.degrees(az) % 360.0


def build_sky(frame_start, frame_end):
    scene = bpy.context.scene
    world = bpy.data.worlds.new('JejuSky')
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    sky = nt.nodes.new('ShaderNodeTexSky')
    for t in ('MULTIPLE_SCATTERING', 'NISHITA', 'SINGLE_SCATTERING'):   # Blender 5.x / 4.x 이름 대응
        try:
            sky.sky_type = t
            break
        except TypeError:
            continue
    sky.sun_disc = True
    sky.sun_size = math.radians(SUN_SIZE_DEG)
    sky.sun_intensity = SUN_INTENSITY
    for attr, val in (('air_density', 1.0), ('dust_density', 0.5), ('aerosol_density', 0.5)):   # 버전별 이름 대응
        if hasattr(sky, attr):
            setattr(sky, attr, val)
    bg = nt.nodes.new('ShaderNodeBackground')
    out = nt.nodes.new('ShaderNodeOutputWorld')
    nt.links.new(sky.outputs[0], bg.inputs['Color'])
    nt.links.new(bg.outputs[0], out.inputs['Surface'])

    # 태양 이동 키프레임 (4프레임 간격, 선형)
    step = 4
    frames = list(range(frame_start, frame_end + 1, step))
    if frames[-1] != frame_end:
        frames.append(frame_end)
    for f in frames:
        t = (f - frame_start) / float(max(1, frame_end - frame_start))
        hour = SUN_START_HOUR + (SUN_END_HOUR - SUN_START_HOUR) * t
        el, az = sun_state(hour)
        sky.sun_elevation = math.radians(el)
        sky.sun_rotation = math.radians(az)        # 방위각: Y(북)에서 X(동) 방향으로 증가 (실측 확인)
        sky.keyframe_insert('sun_elevation', frame=f)
        sky.keyframe_insert('sun_rotation', frame=f)
    make_linear(nt)
    return sky


def build_camera():
    scene = bpy.context.scene
    cam = bpy.data.cameras.new('Camera')
    cam.lens = CAMERA_LENS_MM
    cam.clip_start = 0.5
    cam.clip_end = 50000
    co = bpy.data.objects.new('Camera', cam)
    scene.collection.objects.link(co)
    co.location = CAMERA_POS
    tgt = bpy.data.objects.new('CameraTarget', None)
    tgt.location = CAMERA_TARGET
    scene.collection.objects.link(tgt)
    c = co.constraints.new('TRACK_TO')
    c.target = tgt
    c.track_axis = 'TRACK_NEGATIVE_Z'
    c.up_axis = 'UP_Y'
    scene.camera = co
    return co


def setup_render(opts):
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    cy = sc.cycles
    cy.samples = int(opts.get('samples', SAMPLES))
    cy.use_denoising = True
    try:
        cy.denoiser = 'OPENIMAGEDENOISE'
    except Exception:
        pass
    cy.max_bounces = 12
    cy.transmission_bounces = 12
    cy.volume_bounces = 2
    cy.caustics_reflective = False
    cy.caustics_refractive = False
    cy.sample_clamp_indirect = 10.0
    sc.render.fps = FPS
    sc.frame_start, sc.frame_end = FRAME_START, FRAME_END
    rx, ry = RES_X, RES_Y
    if 'res' in opts and 'x' in str(opts['res']):
        rx, ry = [int(v) for v in str(opts['res']).split('x')]
    sc.render.resolution_x, sc.render.resolution_y = rx, ry
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = 'PNG'
    for vt in ('AgX', 'Filmic'):
        try:
            sc.view_settings.view_transform = vt
            break
        except TypeError:
            continue
    sc.view_settings.exposure = EXPOSURE

    # GPU 사용 시도
    if USE_GPU and 'cpu' not in opts:
        try:
            prefs = bpy.context.preferences.addons['cycles'].preferences
            for dev_type in ('OPTIX', 'CUDA', 'HIP', 'METAL', 'ONEAPI'):
                try:
                    prefs.compute_device_type = dev_type
                    prefs.get_devices()
                    gpus = [d for d in prefs.devices if d.type != 'CPU']
                    if gpus:
                        for d in prefs.devices:
                            d.use = d.type != 'CPU' or False
                        sc.cycles.device = 'GPU'
                        print('[Jeju] GPU 렌더링:', dev_type)
                        break
                except Exception:
                    continue
        except Exception:
            pass


def main():
    opts = parse_args()
    bpy.context.preferences.edit.keyframe_new_interpolation_type = 'LINEAR'
    clear_scene()
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = FRAME_START, FRAME_END
    scene.frame_set(FRAME_START)

    h, nx, ny = make_heightfield()
    terrain, floor = build_terrain(h, nx, ny)
    tmat = make_terrain_material()
    terrain.data.materials.append(tmat)
    floor.data.materials.append(tmat)

    build_ocean(FRAME_START, FRAME_END)
    build_sky(FRAME_START, FRAME_END)
    build_camera()
    setup_render(opts)

    print('[Jeju] 완료!  지형 정점 %d개, 태양: %.1f시→%.1f시, 프레임 %d~%d' % (
        len(terrain.data.vertices), SUN_START_HOUR, SUN_END_HOUR, FRAME_START, FRAME_END))

    # 테스트/배치 렌더
    if 'still' in opts:
        scene.frame_set(int(opts['still']))
        scene.render.filepath = str(opts.get('out', os.path.join(os.path.expanduser('~'), 'jeju_still.png')))
        bpy.ops.render.render(write_still=True)
        print('[Jeju] 저장:', scene.render.filepath)

    if 'save' in opts:
        bpy.ops.wm.save_as_mainfile(filepath=str(opts['save']))
    else:
        try:
            if bpy.data.filepath == '' and '__file__' in globals():
                bpy.ops.wm.save_as_mainfile(filepath=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'jeju_ocean.blend'))
        except Exception:
            pass


main()

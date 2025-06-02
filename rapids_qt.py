from PIL import Image, ImageDraw
import pandas as pd
import numpy as np
import cupy
import cudf
import dask_cudf
import heapq
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from sklearn.metrics.pairwise import cosine_similarity

class Rect:
    def __init__(self, left: int, right: int, top: int, bottom: int):
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        
    # 面積を計算
    def calc_area(self) -> int:
        return (self.right - self.left + 1) * (self.bottom - self.top + 1)
    

class Quad:
    def __init__(self, rect: Rect, score: float, color: tuple):
        self.rect = rect
        self.score = score
        self.color = color
        
    # このメソッドを実装することで、heapqでの優先度付きキューが使える
    # < 演算子の振る舞いを定義するメソッド
    def __lt__(self, other):
        return self.score > other.score
 
    
class QuadTreeGenerator:
    def __init__(
        self,
        arr: cupy.ndarray,
        gene_to_channnel: dict,
        pattern: cudf.DataFrame,
        iterations: int,
        min_size: int,
        area_power: float = 0.2
    ):
        self.arr = arr
        self.gene_to_channel = gene_to_channnel
        self.pattern = pattern
        self.iterations = iterations    # 分割回数
        self.min_size = min_size      # 最小の分割サイズ
        self.area_power = area_power    # 面積の重み
        
        self.gif_frames = []    # GIFのフレーム
        self.canvas = Image.new('RGB', (arr.shape[1], arr.shape[0]))
        self.draw = ImageDraw.Draw(self.canvas)
        root = self.create_quad(
            rect = Rect(0, self.arr.shape[1] - 1, 0, self.arr.shape[0] - 1)
        )
        self.render_and_append_frame(quads = [root])
        self.heap = []  # すべてのQuadを格納する優先度付きキュー
        heapq.heappush(self.heap, root)
    
    
    # quadtreeを生成するメソッド
    def generate_quad_tree(self):
        for _ in range(self.iterations):
            quad = heapq.heappop(self.heap) # ヒープから最もスコアが高いQuadを取り出す
            if(quad.score == 0):
                break
            quads = self.split(quad.rect)    # 取り出したquadを四分割する
            for q in quads:
                heapq.heappush(self.heap, q)    # 四分割したQuadをヒープに追加
            self.render_and_append_frame(quads = quads)
    
    
    # 新しく生成された`Quad`をキャンバスに描画して、`self.gif_frames`に追加するメソッド
    def render_and_append_frame(self, quads: list):
        for quad in quads:
            self.draw.rectangle(
                [
                    (quad.rect.left, quad.rect.top),
                    (quad.rect.right, quad.rect.bottom)
                ],
                fill=quad.color,
                outline=(0, 0, 0),
            )
        self.gif_frames.append(self.canvas.copy())
        
        
    # 指定された`Rect`を元に、scoreを計算して`Quad`を生成するメソッド
    def create_quad(self, rect: Rect) -> Quad:
        score, color = self.calc_score_and_color(rect)
        
        if self.is_quad_below_min_size(rect = rect):
            score = 0
        
        return Quad(rect=rect, score=score, color=color)

    
    def is_quad_below_min_size(self, rect: Rect) -> bool:
        return (
            rect.right - rect.left + 1 <= self.min_size or
            rect.bottom - rect.top + 1 <= self.min_size
        )
    
    # 指定された`Rect`を四分割して生成された、新しい`Quad`のリストを返すメソッド
    def split(self, rect: Rect) -> list:
        # 4つの矩形に分割する
        x_center = (rect.left + rect.right) // 2
        y_center = (rect.top + rect.bottom) // 2
        
        rects = [
            Rect(rect.left, x_center, rect.top, y_center),
            Rect(x_center + 1, rect.right, rect.top, y_center),
            Rect(rect.left, x_center, y_center + 1, rect.bottom),
            Rect(x_center + 1, rect.right, y_center + 1, rect.bottom)
        ]
        
        # 四分割した領域それぞれに対して、Quadを生成
        quads = [self.create_quad(rect) for rect in rects]
        return quads
    
    # scoreを計算するメソッド
    def calc_score_and_color(self, rect: Rect) -> tuple[float, tuple[int, int, int]]:
        V_pre = {}
        V1, V2, V3, V4 = {}, {}, {}, {}

        for gene, channel in self.gene_to_channel.items():
            V_pre[gene] = self.arr[rect.top:rect.bottom+1, rect.left:rect.right+1, channel].sum()
            V1[gene] = self.arr[rect.top:(rect.top+rect.bottom)//2+1, rect.left:(rect.left+rect.right)//2+1, channel].sum()
            V2[gene] = self.arr[rect.top:(rect.top+rect.bottom)//2+1, (rect.left+rect.right)//2+1:rect.right+1, channel].sum()
            V3[gene] = self.arr[(rect.top+rect.bottom)//2+1:rect.bottom+1, rect.left:(rect.left+rect.right)//2+1, channel].sum()
            V4[gene] = self.arr[(rect.top+rect.bottom)//2+1:rect.bottom+1, (rect.left+rect.right)//2+1:rect.right+1, channel].sum()
        
        V_pre = cudf.Series(V_pre).to_frame('expression')
        V_pre = V_pre.reindex(self.pattern.index, fill_value=0)
        V_pre = V_pre.astype(int)
        
        if V_pre['expression'].sum() == 0:
            score = 0
            color = (0, 0, 0)
            return score, color
        
        V1 = cudf.Series(V1).to_frame('expression')
        V1 = V1.reindex(self.pattern.index, fill_value=0)
        V2 = cudf.Series(V2).to_frame('expression')
        V2 = V2.reindex(self.pattern.index, fill_value=0)
        V3 = cudf.Series(V3).to_frame('expression')
        V3 = V3.reindex(self.pattern.index, fill_value=0)
        V4 = cudf.Series(V4).to_frame('expression')
        V4 = V4.reindex(self.pattern.index, fill_value=0)
        
        # 擬似逆行列を計算
        W_inverse = cupy.linalg.pinv(self.pattern.values)
        # 分割前と分割後の係数行列を計算
        H_pre = cupy.dot(W_inverse, V_pre['expression'].values)
        H1 = cupy.dot(W_inverse, V1['expression'].values)
        H2 = cupy.dot(W_inverse, V2['expression'].values)
        H3 = cupy.dot(W_inverse, V3['expression'].values)
        H4 = cupy.dot(W_inverse, V4['expression'].values)
        H_similarities = [
            (cupy.dot(H_pre, H1) / (cupy.linalg.norm(H_pre) * cupy.linalg.norm(H1))).item(),
            (cupy.dot(H_pre, H2) / (cupy.linalg.norm(H_pre) * cupy.linalg.norm(H2))).item(),
            (cupy.dot(H_pre, H3) / (cupy.linalg.norm(H_pre) * cupy.linalg.norm(H3))).item(),
            (cupy.dot(H_pre, H4) / (cupy.linalg.norm(H_pre) * cupy.linalg.norm(H4))).item()
        ]
        
        similarities = cosine_similarity(V_pre.values.get().T, self.pattern.values.get().T)
        similarities = similarities.flatten()
        # 共通する発現パターンに重み付け
        similarities[0] = similarities[0] * 0.8
        similarities[9] = similarities[9] * 0.7
        max_index = np.argmax(similarities)
        match max_index:
            # RGB
            case 0:
                color = (255, 0, 0)   # 赤
            case 1:
                color = (255, 165, 0)   # オレンジ
            case 2:
                color = (255, 255, 0)   # 黄
            case 3:
                color = (50, 205, 50)   # ライム
            case 4:
                color = (0, 128, 0)   # 緑
            case 5:
                color = (0, 255, 255)   # シアン
            case 6:
                color = (0, 0, 255)  # 青
            case 7:
                color = (128, 0, 128)   # 紫
            case 8:
                color = (255, 192, 203)  # ピンク
            case 9:
                color = (255, 255, 255)  # 白色

        if all(sim > 0.9 for sim in H_similarities) and rect.calc_area() <= (self.min_size * 2 * 2) ** 2:
            score = 0
        else:
            # similaritiesを降順にソート
            similarities = np.sort(similarities)[::-1]
            max_index = np.argmax(similarities)
            score = similarities[:3].sum()
        return score, color
    
    
    # `self.gif_frames`のフレームを繋いで、アニメーションGIFを生成するメソッド
    def save_gif(self, output_path: str, duration: int = 30):
        self.gif_frames[0].save(
            output_path,
            save_all=True,
            append_images=self.gif_frames[1:],
            duration=duration,
            loop=1,
        )
        
    def save_png(self, output_path: str):
        frame = self.gif_frames[len(self.gif_frames) - 1]
        frame.save(output_path)
            
    def save_frames_as_png(self, output_dir: str):
        for i, frame in enumerate(self.gif_frames):
            resized_frame = frame.resize((2048, 2048))
            frame.save(f"{output_dir}/frame_{i}.png")
            
            
class QuadExpression:
    def __init__(
        self,
        arr: cupy.ndarray,
        heap, 
        gene_to_channel: dict
    ):
        self.arr = arr
        self.heap = heap
        self.gene_to_channel = gene_to_channel
        
    def calc_expression(self, region: tuple, output_path: str):
        result = cudf.DataFrame()
        for quad in self.heap:
            rect = quad.rect
            gene_expression = {}
            for gene, channel in self.gene_to_channel.items():
                gene_expression[gene] = self.arr[rect.top:rect.bottom+1, rect.left:rect.right+1, channel].sum()
            df = cudf.Series(gene_expression).to_frame()
            df.columns = [f"({region[2]+rect.left},{region[0]+rect.top})_{rect.right-rect.left+1}"]
            #df = df.reindex(self.pattern.index, fill_value=0)
            result = cudf.concat([result, df], axis=1)
        result.to_csv(output_path)
        
        
        
class QuadTreeVisualizer:
    def __init__(self, heap):
        """
        QuadTreeVisualizerクラスの初期化
        :param heap: QuadTreeGenerator から生成されたヒープ (list of Quad)
        """
        self.heap = heap

    def draw_partitions(self, figsize=(10, 10), edge_color='blue', fill=False):
        """
        Quadの境界線を描画する
        :param figsize: 描画する図のサイズ (幅, 高さ)
        :param edge_color: 境界線の色
        :param fill: 領域を塗りつぶすかどうか (True/False)
        """
        
        # 画像の読み込み
        #img = mpimg.imread('../output/subset72/images/mosaic_DAPI_z6_subset72.png')

        # 画像をプロット
        fig, ax = plt.subplots(figsize=(10, 10))

        # 画像を背景として設定
        # extent=[xmin, xmax, ymin, ymax]でプロット範囲を指定
        #ax.imshow(img, extent=[0, img.shape[1], 0, img.shape[0]], origin='lower')
        if len(self.heap) == 0:
            print('No segmentation')
            return
        else:
            for quad in self.heap:
                rect = quad.rect
                width = rect.right - rect.left + 1
                height = rect.bottom - rect.top + 1
                
                # 矩形を追加
                patch = patches.Rectangle(
                    (rect.left, rect.top), width, height,
                    edgecolor=edge_color, facecolor=(edge_color if fill else 'none'), linewidth=1
                )
                ax.add_patch(patch)

            ax.set_xlim(0, max(quad.rect.right for quad in self.heap) + 1)
            ax.set_ylim(0, max(quad.rect.bottom for quad in self.heap) + 1)
            ax.set_aspect('equal')
            plt.gca().invert_yaxis()
            plt.show()
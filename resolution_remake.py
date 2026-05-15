import os
import subprocess
import tempfile
import shutil
import sys


def check_tools():
    """检查必要的工具是否可用"""
    tools = ["ffmpeg", "ffprobe"]
    missing = []

    for tool in tools:
        if not shutil.which(tool):
            missing.append(tool)

    if missing:
        print(f"警告: 以下工具未找到: {', '.join(missing)}")
        print("请安装 ffmpeg:")
        print("  Ubuntu/Debian: sudo apt install ffmpeg")
        print("  macOS: brew install ffmpeg")
        print("  Windows: 下载并添加到PATH")
        return False

    # 检查可选的Real-ESRGAN
    if shutil.which("./realesrgan-ncnn-vulkan/realesrgan-ncnn-vulkan.exe"):
        print("✓ Real-ESRGAN 已安装 (AI超分辨率)")
    else:
        print("! Real-ESRGAN 未找到 (将使用ffmpeg lanczos缩放)")
        print("  安装Real-ESRGAN以获得更好的超分辨率效果:")
        print("  https://github.com/xinntao/Real-ESRGAN")

    return True


def parse_arguments():
    """解析命令行参数"""
    # 默认配置
    input_video = "./videos/20260325150818_40b9302e_chunk_007_5fps.mp4"
    output_video = "./output_sr/super_resolution_output.mp4"
    scale = 2

    # 如果用户提供了命令行参数
    if len(sys.argv) > 1:
        input_video = sys.argv[1]
    if len(sys.argv) > 2:
        output_video = sys.argv[2]
    if len(sys.argv) > 3:
        scale = int(sys.argv[3])

    return input_video, output_video, scale


def validate_inputs(input_video, output_video):
    """验证输入参数是否有效"""
    if not os.path.exists(input_video):
        print(f"错误: 输入视频不存在: {input_video}")
        print("用法: python resolution_remake.py <输入视频> [输出视频] [放大倍数]")
        return False
    return True


def print_config(input_video, output_video, scale):
    """打印配置信息"""
    print(f"输入视频: {input_video}")
    print(f"输出视频: {output_video}")
    print(f"放大倍数: x{scale}")
    print("-" * 60)


def extract_frames(input_video_path, frames_dir):
    """使用ffmpeg提取视频帧"""
    print("[1/4] 提取视频帧...")
    extract_cmd = [
        "ffmpeg", "-i", input_video_path,
        "-q:v", "2",
        os.path.join(frames_dir, "frame_%06d.png")
    ]
    result = subprocess.run(extract_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"提取帧失败: {result.stderr}")
        return None

    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    total_frames = len(frame_files)
    print(f"共提取 {total_frames} 帧")
    return frame_files


def get_video_info(input_video_path):
    """获取视频信息（分辨率、帧率等）"""
    print("[2/4] 获取视频信息...")
    fps = 30  # 默认帧率
    width, height = 0, 0

    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,width,height",
        "-of", "csv=p=0",
        input_video_path
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    if result.returncode == 0:
        info = result.stdout.strip().split(',')
        if len(info) >= 3:
            width, height, fps_str = info[0], info[1], info[2]
            # 处理分数形式的fps (如 "30/1")
            if '/' in fps_str:
                num, den = fps_str.split('/')
                fps = float(num) / float(den)
            else:
                fps = float(fps_str)
            print(f"原视频分辨率: {width}x{height}, FPS: {fps}")

    return fps, int(width), int(height)


def process_frames_realesrgan(frame_files, frames_dir, sr_frames_dir, scale):
    """使用Real-ESRGAN处理帧"""
    realesrgan_exe = "./realesrgan-ncnn-vulkan/realesrgan-ncnn-vulkan.exe"

    # 选择模型
    if scale == 2:
        model_name = "realesrgan-x4plus"
    elif scale == 4:
        model_name = "realesrgan-x4plus"
    else:
        model_name = "realesrgan-x4plus"

    total_frames = len(frame_files)
    for i, frame_file in enumerate(frame_files, 1):
        input_frame = os.path.join(frames_dir, frame_file)
        output_frame = os.path.join(sr_frames_dir, frame_file)

        sr_cmd = [
            realesrgan_exe,
            "-i", input_frame,
            "-o", output_frame,
            "-n", model_name,
            "-s", str(scale)
        ]

        result = subprocess.run(sr_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"处理帧 {frame_file} 失败: {result.stderr}")

        if i % 10 == 0 or i == total_frames:
            print(f"  进度: {i}/{total_frames} ({i/total_frames*100:.1f}%)")


def process_frames_ffmpeg(frame_files, frames_dir, sr_frames_dir, scale):
    """使用ffmpeg lanczos算法处理帧（备选方案）"""
    total_frames = len(frame_files)
    for i, frame_file in enumerate(frame_files, 1):
        input_frame = os.path.join(frames_dir, frame_file)
        output_frame = os.path.join(sr_frames_dir, frame_file)

        sr_cmd = [
            "ffmpeg", "-y", "-i", input_frame,
            "-vf", f"scale=iw*{scale}:ih*{scale}:flags=lanczos",
            "-q:v", "2",
            output_frame
        ]

        result = subprocess.run(sr_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"处理帧 {frame_file} 失败: {result.stderr}")

        if i % 10 == 0 or i == total_frames:
            print(f"  进度: {i}/{total_frames} ({i/total_frames*100:.1f}%)")


def apply_super_resolution(frame_files, frames_dir, sr_frames_dir, scale):
    """应用超分辨率处理"""
    print(f"[3/4] 超分辨率重构 (x{scale})...")

    realesrgan_exe = "realesrgan-ncnn-vulkan"

    if shutil.which(realesrgan_exe):
        print(f"使用 {realesrgan_exe} 进行超分辨率...")
        process_frames_realesrgan(frame_files, frames_dir, sr_frames_dir, scale)
    else:
        print(f"{realesrgan_exe} 未找到，使用ffmpeg lanczos缩放...")
        process_frames_ffmpeg(frame_files, frames_dir, sr_frames_dir, scale)


def encode_video(sr_frames_dir, output_video_path, input_video_path, fps):
    """使用ffmpeg合成视频"""
    print("[4/4] 合成视频...")

    # 确保输出目录存在
    output_dir = os.path.dirname(os.path.abspath(output_video_path))
    os.makedirs(output_dir, exist_ok=True)

    encode_cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(sr_frames_dir, "frame_%06d.png"),
        "-i", input_video_path,
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "slow",
        "-c:a", "copy",
        "-pix_fmt", "yuv420p",
        output_video_path
    ]

    result = subprocess.run(encode_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"合成视频失败: {result.stderr}")
        return False

    print(f"超分辨率视频已保存: {output_video_path}")

    # 计算输出视频信息
    if os.path.exists(output_video_path):
        output_size = os.path.getsize(output_video_path) / (1024 * 1024)
        print(f"输出文件大小: {output_size:.2f} MB")

    return True


def cleanup(temp_dir):
    """清理临时文件"""
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"清理临时文件: {temp_dir}")


def process_video_super_resolution(input_video_path, output_video_path, scale=2):
    """
    主处理流程：视频超分辨率重构

    Args:
        input_video_path: 输入视频路径
        output_video_path: 输出视频路径
        scale: 放大倍数

    Returns:
        bool: 是否成功
    """
    if not os.path.exists(input_video_path):
        print(f"错误: 输入视频不存在: {input_video_path}")
        return False

    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix="sr_frames_")
    frames_dir = os.path.join(temp_dir, "frames")
    sr_frames_dir = os.path.join(temp_dir, "sr_frames")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(sr_frames_dir, exist_ok=True)

    try:
        # 步骤1: 提取帧
        frame_files = extract_frames(input_video_path, frames_dir)
        if not frame_files:
            return False

        # 步骤2: 获取视频信息
        fps, _, _ = get_video_info(input_video_path)

        # 步骤3: 超分辨率处理
        apply_super_resolution(frame_files, frames_dir, sr_frames_dir, scale)

        # 步骤4: 合成视频
        success = encode_video(sr_frames_dir, output_video_path, input_video_path, fps)

        return success

    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        cleanup(temp_dir)


def main():
    """主函数：实现视频超分辨率重构+保存"""
    print("=" * 60)
    print("视频超分辨率重构工具")
    print("=" * 60)

    # 检查工具
    if not check_tools():
        return

    # 解析命令行参数
    input_video, output_video, scale = parse_arguments()

    # 验证输入
    if not validate_inputs(input_video, output_video):
        return

    # 打印配置
    print_config(input_video, output_video, scale)

    # 执行超分辨率处理
    success = process_video_super_resolution(input_video, output_video, scale)

    # 打印结果
    print("=" * 60)
    if success:
        print("处理完成!")
    else:
        print("处理失败!")
    print("=" * 60)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
新架构API提供商测试脚本
测试配置管理器和API客户端的基本功能
"""

import asyncio
import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config_manager import get_new_config_manager
from core.new_api_client import get_text_client, get_vision_client, get_generate_client, close_all_clients
from loguru import logger

async def test_config_manager():
    """测试配置管理器"""
    print("=" * 50)
    print("🔧 测试配置管理器")
    print("=" * 50)

    try:
        # 初始化配置管理器
        config_manager = get_new_config_manager()

        # 检查各功能配置状态
        print(f"✓ 配置管理器初始化成功")

        # 文本理解配置
        text_config = config_manager.get_text_config()
        print(f"📝 文本理解配置:")
        print(f"   Primary: {'✓' if text_config.has_primary() else '✗'}")
        print(f"   Backup: {'✓' if text_config.has_backup() else '✗'}")
        if text_config.has_primary():
            print(f"   Primary Model: {text_config.primary_model}")
            print(f"   Primary RPM: {text_config.primary_rpm}")

        # 多模态理解配置
        vision_config = config_manager.get_vision_config()
        print(f"👁 多模态理解配置:")
        print(f"   Primary: {'✓' if vision_config.has_primary() else '✗'}")
        print(f"   Backup: {'✓' if vision_config.has_backup() else '✗'}")
        if vision_config.has_primary():
            print(f"   Primary Model: {vision_config.primary_model}")
            print(f"   Primary RPM: {vision_config.primary_rpm}")

        # LLM生成配置
        generate_config = config_manager.get_generate_config()
        print(f"🤖 LLM生成配置:")
        print(f"   Primary: {'✓' if generate_config.has_primary() else '✗'}")
        print(f"   Backup: {'✓' if generate_config.has_backup() else '✗'}")
        if generate_config.has_primary():
            print(f"   Primary Model: {generate_config.primary_model}")
            print(f"   Primary RPM: {generate_config.primary_rpm}")

        print("✅ 配置管理器测试通过")
        return True

    except Exception as e:
        print(f"❌ 配置管理器测试失败: {e}")
        return False

async def test_client_initialization():
    """测试客户端初始化"""
    print("\n" + "=" * 50)
    print("🔌 测试API客户端初始化")
    print("=" * 50)

    try:
        # 测试文本理解客户端
        print("📝 初始化文本理解客户端...")
        text_client = await get_text_client()
        print(f"   ✓ 文本理解客户端初始化成功")

        # 测试多模态理解客户端
        print("👁 初始化多模态理解客户端...")
        vision_client = await get_vision_client()
        print(f"   ✓ 多模态理解客户端初始化成功")

        # 测试LLM生成客户端
        print("🤖 初始化LLM生成客户端...")
        generate_client = await get_generate_client()
        print(f"   ✓ LLM生成客户端初始化成功")

        print("✅ 客户端初始化测试通过")
        return True

    except Exception as e:
        print(f"❌ 客户端初始化测试失败: {e}")
        return False

async def test_api_calls():
    """测试API调用（不发送真实请求，只测试调用链路）"""
    print("\n" + "=" * 50)
    print("📡 测试API调用链路")
    print("=" * 50)

    try:
        config_manager = get_new_config_manager()

        # 测试文本理解调用链路
        if config_manager.is_function_available("text"):
            print("📝 测试文本理解调用链路...")
            text_client = await get_text_client()

            # 模拟调用参数
            test_kwargs = {
                "prompt": "测试文本理解",
                "system_prompt": "You are a test assistant",
                "temperature": 0.1,
                "max_tokens": 100
            }

            print("   ✓ 文本理解调用链路准备就绪")
        else:
            print("   ⚠ 文本理解功能未配置，跳过测试")

        # 测试多模态理解调用链路
        if config_manager.is_function_available("vision"):
            print("👁 测试多模态理解调用链路...")
            vision_client = await get_vision_client()
            print("   ✓ 多模态理解调用链路准备就绪")
        else:
            print("   ⚠ 多模态理解功能未配置，跳过测试")

        # 测试LLM生成调用链路
        if config_manager.is_function_available("generate"):
            print("🤖 测试LLM生成调用链路...")
            generate_client = await get_generate_client()
            print("   ✓ LLM生成调用链路准备就绪")
        else:
            print("   ⚠ LLM生成功能未配置，跳过测试")

        print("✅ API调用链路测试通过")
        return True

    except Exception as e:
        print(f"❌ API调用链路测试失败: {e}")
        return False

async def test_rate_limiters():
    """测试速率限制器"""
    print("\n" + "=" * 50)
    print("⏱ 测试速率限制器")
    print("=" * 50)

    try:
        from core.new_api_client import FunctionRateLimiter

        # 创建测试速率限制器
        limiter = FunctionRateLimiter(rpm_limit=60, min_interval=0.01)

        print("⏱ 测试速率限制器初始化...")
        print(f"   RPM限制: {limiter.rpm_limit}")
        print(f"   最小间隔: {limiter.min_interval}秒")

        # 模拟快速调用
        import time
        start_time = time.time()

        print("⏱ 模拟连续API调用...")
        for i in range(3):
            await limiter.wait_if_needed()
            print(f"   调用 {i+1} 完成")

        elapsed = time.time() - start_time
        print(f"   总耗时: {elapsed:.3f}秒")

        print("✅ 速率限制器测试通过")
        return True

    except Exception as e:
        print(f"❌ 速率限制器测试失败: {e}")
        return False

async def test_cleanup():
    """测试资源清理"""
    print("\n" + "=" * 50)
    print("🧹 测试资源清理")
    print("=" * 50)

    try:
        await close_all_clients()
        print("✅ 客户端资源清理完成")
        return True

    except Exception as e:
        print(f"❌ 资源清理失败: {e}")
        return False

async def main():
    """主测试函数"""
    print("🧪 FinanceRAG-Pro 新架构测试")
    print("测试配置管理器和API客户端架构")
    print("=" * 60)

    # 检查环境文件
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠ 警告: .env文件不存在，将使用默认配置")
        print("请根据需要复制 configs/.env.example 到 .env 并配置API密钥")
        print()

    test_results = []

    # 执行测试
    test_results.append(await test_config_manager())
    test_results.append(await test_client_initialization())
    test_results.append(await test_api_calls())
    test_results.append(await test_rate_limiters())
    test_results.append(await test_cleanup())

    # 汇总结果
    print("\n" + "=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)

    passed = sum(test_results)
    total = len(test_results)

    print(f"通过测试: {passed}/{total}")
    print(f"成功率: {passed/total*100:.1f}%")

    if passed == total:
        print("🎉 所有测试通过！新架构运行正常")
        return 0
    else:
        print("⚠ 部分测试失败，请检查配置和实现")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
#!/usr/bin/env python
"""
Quick Test Script for FinanceRAG-Pro
快速测试系统各组件功能的脚本
"""

import asyncio
import sys
import os
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any
import httpx
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

console = Console()

class QuickTester:
    """快速测试器"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
        self.results = []
        
    async def test_service_health(self) -> Dict[str, Any]:
        """测试服务健康状态"""
        try:
            start_time = time.time()
            response = await self.client.get(f"{self.base_url}/health")
            end_time = time.time()
            
            success = response.status_code == 200
            return {
                "test": "Service Health Check",
                "success": success,
                "response_time": end_time - start_time,
                "status_code": response.status_code,
                "details": response.json() if success else response.text
            }
        except Exception as e:
            return {
                "test": "Service Health Check",
                "success": False,
                "error": str(e),
                "details": "Service may not be running"
            }
    
    async def test_api_docs(self) -> Dict[str, Any]:
        """测试API文档可访问性"""
        try:
            start_time = time.time()
            response = await self.client.get(f"{self.base_url}/docs")
            end_time = time.time()
            
            success = response.status_code == 200
            return {
                "test": "API Documentation",
                "success": success,
                "response_time": end_time - start_time,
                "status_code": response.status_code,
                "details": "Swagger UI accessible" if success else response.text[:100]
            }
        except Exception as e:
            return {
                "test": "API Documentation",
                "success": False,
                "error": str(e)
            }
    
    async def test_simple_query(self) -> Dict[str, Any]:
        """测试简单查询功能"""
        try:
            query_data = {
                "query": "测试查询：什么是财务报表？",
                "top_k": 5
            }
            
            start_time = time.time()
            response = await self.client.post(f"{self.base_url}/query", json=query_data)
            end_time = time.time()
            
            success = response.status_code == 200
            result_data = response.json() if success else {}
            
            return {
                "test": "Simple Query Processing",
                "success": success,
                "response_time": end_time - start_time,
                "status_code": response.status_code,
                "details": {
                    "has_answer": "answer" in result_data,
                    "has_confidence": "confidence" in result_data,
                    "answer_length": len(result_data.get("answer", "")),
                } if success else response.text[:200]
            }
        except Exception as e:
            return {
                "test": "Simple Query Processing",
                "success": False,
                "error": str(e)
            }
    
    async def test_file_upload(self) -> Dict[str, Any]:
        """测试文件上传功能"""
        try:
            # Create a minimal test PDF
            test_content = b"%PDF-1.4\n%Test PDF for FinanceRAG\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n>>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000010 00000 n \n0000000068 00000 n \n0000000125 00000 n \ntrailer\n<<\n/Size 4\n/Root 1 0 R\n>>\nstartxref\n196\n%%EOF"
            
            files = {
                'file': ('test_report.pdf', test_content, 'application/pdf')
            }
            
            start_time = time.time()
            response = await self.client.post(f"{self.base_url}/upload", files=files)
            end_time = time.time()
            
            success = response.status_code == 200
            result_data = response.json() if success else {}
            
            return {
                "test": "File Upload",
                "success": success,
                "response_time": end_time - start_time,
                "status_code": response.status_code,
                "details": {
                    "document_id": result_data.get("document_id"),
                    "status": result_data.get("status"),
                } if success else response.text[:200]
            }
        except Exception as e:
            return {
                "test": "File Upload",
                "success": False,
                "error": str(e)
            }
    
    async def test_invalid_requests(self) -> Dict[str, Any]:
        """测试无效请求处理"""
        tests = []
        
        # Test empty query
        try:
            response = await self.client.post(f"{self.base_url}/query", json={"query": "", "top_k": 5})
            tests.append(("Empty Query", response.status_code in [400, 422]))
        except:
            tests.append(("Empty Query", False))
        
        # Test invalid JSON
        try:
            response = await self.client.post(
                f"{self.base_url}/query",
                content="invalid json",
                headers={"content-type": "application/json"}
            )
            tests.append(("Invalid JSON", response.status_code == 422))
        except:
            tests.append(("Invalid JSON", False))
        
        # Test missing fields
        try:
            response = await self.client.post(f"{self.base_url}/query", json={"top_k": 5})
            tests.append(("Missing Fields", response.status_code == 422))
        except:
            tests.append(("Missing Fields", False))
        
        passed = sum(1 for _, success in tests if success)
        total = len(tests)
        
        return {
            "test": "Invalid Request Handling",
            "success": passed == total,
            "details": f"Passed {passed}/{total} validation tests",
            "breakdown": tests
        }
    
    async def test_performance_baseline(self) -> Dict[str, Any]:
        """测试基本性能指标"""
        try:
            # Test multiple quick requests
            query_data = {
                "query": "性能测试查询",
                "top_k": 3
            }
            
            times = []
            successes = 0
            
            for i in range(5):
                try:
                    start_time = time.time()
                    response = await self.client.post(f"{self.base_url}/query", json=query_data)
                    end_time = time.time()
                    
                    times.append(end_time - start_time)
                    if response.status_code == 200:
                        successes += 1
                except:
                    pass
            
            if times:
                avg_time = sum(times) / len(times)
                max_time = max(times)
                min_time = min(times)
                
                return {
                    "test": "Performance Baseline",
                    "success": successes >= 3,  # At least 3/5 should succeed
                    "details": {
                        "successful_requests": f"{successes}/5",
                        "avg_response_time": f"{avg_time:.3f}s",
                        "min_response_time": f"{min_time:.3f}s",
                        "max_response_time": f"{max_time:.3f}s",
                        "performance_rating": "Good" if avg_time < 2.0 else "Acceptable" if avg_time < 5.0 else "Slow"
                    }
                }
            else:
                return {
                    "test": "Performance Baseline",
                    "success": False,
                    "details": "No successful requests"
                }
        except Exception as e:
            return {
                "test": "Performance Baseline",
                "success": False,
                "error": str(e)
            }
    
    async def run_all_tests(self) -> List[Dict[str, Any]]:
        """运行所有测试"""
        console.print(Panel(
            "[bold cyan]FinanceRAG-Pro 快速测试[/bold cyan]\n"
            "检查系统各组件基本功能",
            title="🧪 Quick Test Suite"
        ))
        
        tests = [
            ("检查服务状态", self.test_service_health),
            ("检查API文档", self.test_api_docs),
            ("测试简单查询", self.test_simple_query),
            ("测试文件上传", self.test_file_upload),
            ("测试错误处理", self.test_invalid_requests),
            ("性能基准测试", self.test_performance_baseline),
        ]
        
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            
            task = progress.add_task("运行测试...", total=len(tests))
            
            for test_name, test_func in tests:
                progress.update(task, description=f"运行: {test_name}")
                
                try:
                    result = await test_func()
                    results.append(result)
                    
                    status = "[green]✓[/green]" if result["success"] else "[red]✗[/red]"
                    console.print(f"{status} {test_name}")
                    
                except Exception as e:
                    error_result = {
                        "test": test_name,
                        "success": False,
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    }
                    results.append(error_result)
                    console.print(f"[red]✗[/red] {test_name} - {str(e)}")
                
                progress.advance(task)
        
        return results
    
    def display_results(self, results: List[Dict[str, Any]]):
        """显示测试结果"""
        # Summary table
        table = Table(title="测试结果汇总")
        table.add_column("测试项目", style="cyan")
        table.add_column("状态", style="bold")
        table.add_column("响应时间", style="yellow")
        table.add_column("详情", style="dim")
        
        passed = 0
        total = len(results)
        
        for result in results:
            status = "[green]PASS[/green]" if result["success"] else "[red]FAIL[/red]"
            response_time = f"{result.get('response_time', 0):.3f}s" if 'response_time' in result else "N/A"
            
            details = ""
            if result["success"]:
                passed += 1
                if isinstance(result.get("details"), dict):
                    details = str(result["details"])[:50] + "..."
                else:
                    details = str(result.get("details", ""))[:50]
            else:
                details = result.get("error", "Unknown error")[:50]
            
            table.add_row(
                result["test"],
                status,
                response_time,
                details
            )
        
        console.print(table)
        
        # Summary panel
        success_rate = (passed / total) * 100 if total > 0 else 0
        color = "green" if success_rate >= 80 else "yellow" if success_rate >= 60 else "red"
        
        summary = f"""
[bold]测试完成！[/bold]

通过: {passed}/{total} ({success_rate:.1f}%)
        
[{color}]系统状态: {"优秀" if success_rate >= 90 else "良好" if success_rate >= 80 else "需要关注" if success_rate >= 60 else "存在问题"}[/{color}]
        """
        
        console.print(Panel(summary, title="📊 测试汇总", border_style=color))
        
        # Detailed results for failed tests
        failed_tests = [r for r in results if not r["success"]]
        if failed_tests:
            console.print("\n[red]失败的测试详情：[/red]")
            for result in failed_tests:
                console.print(f"\n[bold red]❌ {result['test']}[/bold red]")
                if 'error' in result:
                    console.print(f"   错误: {result['error']}")
                if 'details' in result:
                    console.print(f"   详情: {result['details']}")
        
        return success_rate
    
    async def close(self):
        """关闭客户端"""
        await self.client.aclose()


# 独立功能测试
def test_imports():
    """测试模块导入"""
    try:
        console.print("[yellow]测试模块导入...[/yellow]")
        
        # Test core modules
        from core.document_processor import MultiModalPreprocessor
        from core.retrieval_engine import MultiPathRetrievalEngine
        from core.query_engine import QueryRoutingEngine
        from core.validation_engine import ValidationEngine
        from core.inference_engine import InferenceEngine
        
        console.print("[green]✓[/green] 所有核心模块导入成功")
        return True
        
    except Exception as e:
        console.print(f"[red]✗[/red] 模块导入失败: {e}")
        return False

def test_configuration():
    """测试配置文件"""
    try:
        console.print("[yellow]测试配置文件...[/yellow]")
        
        config_files = ['config_lite.yaml', 'config.yaml', 'config_api.yaml']
        found_configs = []
        
        for config_file in config_files:
            if Path(config_file).exists():
                found_configs.append(config_file)
        
        if found_configs:
            console.print(f"[green]✓[/green] 找到配置文件: {', '.join(found_configs)}")
            return True
        else:
            console.print("[red]✗[/red] 未找到配置文件")
            return False
            
    except Exception as e:
        console.print(f"[red]✗[/red] 配置测试失败: {e}")
        return False

def test_directories():
    """测试目录结构"""
    try:
        console.print("[yellow]测试目录结构...[/yellow]")
        
        required_dirs = ['data', 'logs', 'tests', 'core', 'config']
        missing_dirs = []
        
        for directory in required_dirs:
            if not Path(directory).exists():
                missing_dirs.append(directory)
        
        if not missing_dirs:
            console.print("[green]✓[/green] 所有必要目录存在")
            return True
        else:
            console.print(f"[yellow]⚠[/yellow] 缺少目录: {', '.join(missing_dirs)}")
            # Create missing directories
            for directory in missing_dirs:
                Path(directory).mkdir(parents=True, exist_ok=True)
            console.print("[green]✓[/green] 已创建缺失目录")
            return True
            
    except Exception as e:
        console.print(f"[red]✗[/red] 目录检查失败: {e}")
        return False


async def main():
    """主函数"""
    console.print("""
[bold cyan]╔══════════════════════════════════════════╗
║        FinanceRAG-Pro 快速测试工具       ║
║         Quick Test & Health Check        ║
╚══════════════════════════════════════════╝[/bold cyan]
    """)
    
    # 选择测试模式
    console.print("选择测试模式:")
    console.print("1. 快速检查 (离线测试)")
    console.print("2. 完整测试 (需要服务运行)")
    console.print("3. 仅服务连接测试")
    console.print("4. 退出")
    
    choice = input("\n请输入选择 (1-4): ").strip()
    
    if choice == "1":
        # 离线测试
        console.print("\n[cyan]执行离线快速检查...[/cyan]")
        
        results = []
        results.append(test_imports())
        results.append(test_configuration())
        results.append(test_directories())
        
        passed = sum(results)
        total = len(results)
        
        console.print(f"\n[bold]离线检查完成: {passed}/{total} 通过[/bold]")
        
        if passed == total:
            console.print("[green]✅ 系统准备就绪，可以启动服务进行完整测试[/green]")
        else:
            console.print("[yellow]⚠️ 系统存在配置问题，请检查后再试[/yellow]")
    
    elif choice == "2":
        # 完整测试
        tester = QuickTester()
        try:
            results = await tester.run_all_tests()
            success_rate = tester.display_results(results)
            
            if success_rate >= 80:
                console.print("\n[green]🎉 系统运行良好，可以进行演示！[/green]")
            else:
                console.print("\n[yellow]⚠️ 系统存在问题，建议检查后再进行演示[/yellow]")
                
        finally:
            await tester.close()
    
    elif choice == "3":
        # 仅连接测试
        tester = QuickTester()
        try:
            console.print("\n[cyan]测试服务连接...[/cyan]")
            result = await tester.test_service_health()
            
            if result["success"]:
                console.print("[green]✅ 服务连接成功！[/green]")
                console.print(f"响应时间: {result.get('response_time', 0):.3f}秒")
            else:
                console.print("[red]❌ 服务连接失败[/red]")
                console.print(f"错误: {result.get('error', 'Unknown')}")
                
        finally:
            await tester.close()
    
    elif choice == "4":
        console.print("[yellow]退出测试[/yellow]")
    
    else:
        console.print("[red]无效选择[/red]")


if __name__ == "__main__":
    # 检查依赖
    try:
        import rich
        import httpx
    except ImportError as e:
        print(f"缺少依赖: {e}")
        print("请运行: pip install rich httpx")
        sys.exit(1)
    
    # 运行测试
    asyncio.run(main())
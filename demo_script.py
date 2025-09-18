#!/usr/bin/env python
"""
FinanceRAG-Pro Demo Script
演示从PDF上传到查询的完整流程
"""

import asyncio
import sys
import os
import time
import json
from pathlib import Path
from typing import List, Dict, Any
import httpx
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.markdown import Markdown

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

console = Console()

class FinanceRAGDemo:
    """FinanceRAG-Pro演示脚本"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60.0)
        self.uploaded_files = []
        
    async def check_health(self) -> bool:
        """检查服务健康状态"""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception as e:
            console.print(f"[red]健康检查失败: {e}[/red]")
            return False
    
    async def upload_pdf(self, file_path: str) -> Dict[str, Any]:
        """上传PDF文档"""
        with open(file_path, 'rb') as f:
            files = {'file': (Path(file_path).name, f, 'application/pdf')}
            response = await self.client.post(
                f"{self.base_url}/upload",
                files=files
            )
        
        if response.status_code == 200:
            result = response.json()
            self.uploaded_files.append(result)
            return result
        else:
            raise Exception(f"上传失败: {response.text}")
    
    async def query(self, question: str, top_k: int = 10) -> Dict[str, Any]:
        """执行查询"""
        payload = {
            "query": question,
            "top_k": top_k
        }
        
        response = await self.client.post(
            f"{self.base_url}/query",
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"查询失败: {response.text}")
    
    async def run_demo_scenario(self, scenario: Dict[str, Any]):
        """运行单个演示场景"""
        console.print(Panel(f"[bold blue]{scenario['title']}[/bold blue]"))
        console.print(f"\n📄 文档: {scenario['file']}")
        console.print(f"❓ 问题: {scenario['question']}\n")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("处理查询...", total=None)
            
            start_time = time.time()
            result = await self.query(scenario['question'])
            end_time = time.time()
            
            progress.stop()
        
        # 显示结果
        console.print(f"[green]✅ 查询完成 (耗时: {end_time - start_time:.2f}秒)[/green]\n")
        
        # 显示答案
        console.print(Panel(
            Markdown(f"**答案:**\n\n{result['answer']}"),
            title="🎯 系统回答",
            border_style="green"
        ))
        
        # 显示置信度和来源
        console.print(f"\n📊 置信度: {result['confidence']:.2%}")
        
        if result.get('sources'):
            console.print("\n📚 信息来源:")
            for source in result['sources'][:3]:  # 只显示前3个来源
                console.print(f"  - {source}")
        
        console.print("\n" + "="*60 + "\n")
        
        return result
    
    async def run_batch_demo(self, demo_cases: List[Dict[str, Any]]):
        """运行批量演示"""
        console.print(Panel(
            "[bold cyan]FinanceRAG-Pro 演示系统[/bold cyan]\n"
            "展示从真实财报PDF自动生成答案的能力",
            title="🚀 系统演示",
            border_style="cyan"
        ))
        
        # 检查服务状态
        console.print("\n[yellow]检查服务状态...[/yellow]")
        if not await self.check_health():
            console.print("[red]❌ 服务未启动，请先运行: python main_lite.py[/red]")
            return
        
        console.print("[green]✅ 服务正常运行[/green]\n")
        
        # 上传演示文档
        console.print("[yellow]上传演示文档...[/yellow]")
        unique_files = list(set(case['file'] for case in demo_cases))
        
        for file_path in unique_files:
            if Path(file_path).exists():
                try:
                    result = await self.upload_pdf(file_path)
                    console.print(f"[green]✅ 已上传: {Path(file_path).name}[/green]")
                except Exception as e:
                    console.print(f"[red]❌ 上传失败 {file_path}: {e}[/red]")
            else:
                console.print(f"[yellow]⚠️ 文件不存在: {file_path}[/yellow]")
        
        console.print("\n" + "="*60 + "\n")
        
        # 运行演示场景
        results = []
        for i, scenario in enumerate(demo_cases, 1):
            console.print(f"[cyan]场景 {i}/{len(demo_cases)}[/cyan]")
            try:
                result = await self.run_demo_scenario(scenario)
                results.append({
                    "scenario": scenario,
                    "result": result,
                    "success": True
                })
            except Exception as e:
                console.print(f"[red]❌ 场景执行失败: {e}[/red]\n")
                results.append({
                    "scenario": scenario,
                    "error": str(e),
                    "success": False
                })
        
        # 显示汇总结果
        self.show_summary(results)
        
        return results
    
    def show_summary(self, results: List[Dict[str, Any]]):
        """显示汇总结果"""
        console.print(Panel(
            "[bold green]演示完成汇总[/bold green]",
            border_style="green"
        ))
        
        table = Table(title="执行结果统计")
        table.add_column("指标", style="cyan")
        table.add_column("数值", style="green")
        
        total = len(results)
        success = sum(1 for r in results if r['success'])
        failed = total - success
        
        table.add_row("总场景数", str(total))
        table.add_row("成功", str(success))
        table.add_row("失败", str(failed))
        table.add_row("成功率", f"{success/total*100:.1f}%")
        
        console.print(table)
        
        # 保存结果到文件
        output_file = f"demo_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        console.print(f"\n📁 详细结果已保存到: {output_file}")
    
    async def close(self):
        """关闭客户端连接"""
        await self.client.aclose()


# 预定义的演示场景
DEMO_CASES = [
    # Level 1: 基础事实查询
    {
        "title": "财务数据查询",
        "file": "data/pdfs/广联达2022年半年报.pdf",
        "question": "广联达公司2022年上半年营业收入增长情况如何？",
        "expected_type": "数字和增长率"
    },
    {
        "title": "利润指标查询",
        "file": "data/pdfs/千味央厨2020年年报.pdf",  
        "question": "千味央厨（001215.SZ）在2020年的营业收入和净利润分别占总营收和净利润的比例是多少？",
        "expected_type": "比例数据"
    },
    
    # Level 2: 业务分析查询
    {
        "title": "战略转型分析",
        "file": "data/pdfs/广联达2022年半年报.pdf",
        "question": "广联达公司如何通过云转型和多元化业务布局实现业绩增长？",
        "expected_type": "策略描述和成果"
    },
    {
        "title": "竞争优势分析",
        "file": "data/pdfs/千味央厨2020年年报.pdf",
        "question": "千味央厨在速冻食品行业的发展中，具体有哪些优势？",
        "expected_type": "优势列表"
    },
    
    # Level 3: 复杂综合查询
    {
        "title": "技术架构分析",
        "file": "data/pdfs/中恒电气2024年报告.pdf",
        "question": "中恒电气的数字项目集成管理平台总体架构如何？",
        "expected_type": "架构描述"
    }
]


async def main():
    """主函数"""
    console.print("""
[bold cyan]╔════════════════════════════════════════╗
║     FinanceRAG-Pro 演示脚本 v1.0      ║
║    金融多模态智能检索系统演示          ║
╚════════════════════════════════════════╝[/bold cyan]
    """)
    
    # 选择演示模式
    console.print("\n请选择演示模式:")
    console.print("1. 快速演示 (3个场景)")
    console.print("2. 完整演示 (5个场景)")
    console.print("3. 自定义查询")
    console.print("4. 退出")
    
    choice = input("\n请输入选择 (1-4): ")
    
    demo = FinanceRAGDemo()
    
    try:
        if choice == "1":
            # 快速演示
            await demo.run_batch_demo(DEMO_CASES[:3])
        
        elif choice == "2":
            # 完整演示
            await demo.run_batch_demo(DEMO_CASES)
        
        elif choice == "3":
            # 自定义查询
            console.print("\n[cyan]自定义查询模式[/cyan]")
            console.print("输入 'quit' 退出\n")
            
            while True:
                question = input("请输入查询问题: ")
                if question.lower() == 'quit':
                    break
                
                try:
                    with console.status("[bold green]处理中..."):
                        result = await demo.query(question)
                    
                    console.print(Panel(
                        Markdown(f"**答案:**\n\n{result['answer']}"),
                        title="回答",
                        border_style="green"
                    ))
                    console.print(f"置信度: {result['confidence']:.2%}\n")
                    
                except Exception as e:
                    console.print(f"[red]查询失败: {e}[/red]")
        
        elif choice == "4":
            console.print("[yellow]退出演示[/yellow]")
        
        else:
            console.print("[red]无效的选择[/red]")
    
    finally:
        await demo.close()
        console.print("\n[green]演示结束，感谢使用！[/green]")


if __name__ == "__main__":
    # 检查依赖
    try:
        import rich
    except ImportError:
        print("请先安装依赖: pip install rich httpx")
        sys.exit(1)
    
    # 运行演示
    asyncio.run(main())
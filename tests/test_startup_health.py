"""
Tests for startup and health monitoring functionality
"""
import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime

from app.core.startup import StartupManager, startup_manager
from app.core.health import HealthMonitor, HealthStatus, ComponentHealth, SystemMetrics
from config.settings import Settings


class TestStartupManager:
    """Test startup manager functionality"""
    
    def test_startup_manager_initialization(self):
        """Test startup manager initialization"""
        manager = StartupManager()
        
        assert manager.startup_tasks == []
        assert manager.shutdown_tasks == []
        assert manager.startup_completed is False
        assert manager.startup_errors == []
        assert manager.startup_start_time is None
    
    def test_add_startup_task(self):
        """Test adding startup tasks"""
        manager = StartupManager()
        
        async def test_task():
            pass
        
        manager.add_startup_task("test_task", test_task, critical=True, timeout=30)
        
        assert len(manager.startup_tasks) == 1
        assert manager.startup_tasks[0]['name'] == "test_task"
        assert manager.startup_tasks[0]['func'] == test_task
        assert manager.startup_tasks[0]['critical'] is True
        assert manager.startup_tasks[0]['timeout'] == 30
    
    def test_add_shutdown_task(self):
        """Test adding shutdown tasks"""
        manager = StartupManager()
        
        async def test_task():
            pass
        
        manager.add_shutdown_task("test_task", test_task, timeout=10)
        
        assert len(manager.shutdown_tasks) == 1
        assert manager.shutdown_tasks[0]['name'] == "test_task"
        assert manager.shutdown_tasks[0]['func'] == test_task
        assert manager.shutdown_tasks[0]['timeout'] == 10
    
    @pytest.mark.asyncio
    async def test_successful_startup_sequence(self):
        """Test successful startup sequence"""
        manager = StartupManager()
        
        # Mock successful tasks
        task1_called = False
        task2_called = False
        
        async def task1():
            nonlocal task1_called
            task1_called = True
        
        async def task2():
            nonlocal task2_called
            task2_called = True
        
        manager.add_startup_task("task1", task1)
        manager.add_startup_task("task2", task2)
        
        # Mock configuration and validation
        with patch('app.core.startup.initialize_configuration'), \
             patch('app.core.startup.validate_configuration', return_value=True), \
             patch.object(manager, '_start_model_preloading'):
            
            result = await manager.execute_startup_sequence()
            
            assert result is True
            assert manager.startup_completed is True
            assert task1_called is True
            assert task2_called is True
            assert len(manager.startup_errors) == 0
    
    @pytest.mark.asyncio
    async def test_failed_startup_sequence(self):
        """Test failed startup sequence"""
        manager = StartupManager()
        
        async def failing_task():
            raise RuntimeError("Task failed")
        
        manager.add_startup_task("failing_task", failing_task, critical=True)
        
        # Mock configuration and validation
        with patch('app.core.startup.initialize_configuration'), \
             patch('app.core.startup.validate_configuration', return_value=True):
            
            result = await manager.execute_startup_sequence()
            
            assert result is False
            assert manager.startup_completed is False
            assert len(manager.startup_errors) > 0
    
    @pytest.mark.asyncio
    async def test_non_critical_task_failure(self):
        """Test non-critical task failure doesn't stop startup"""
        manager = StartupManager()
        
        success_called = False
        
        async def failing_task():
            raise RuntimeError("Non-critical task failed")
        
        async def success_task():
            nonlocal success_called
            success_called = True
        
        manager.add_startup_task("failing_task", failing_task, critical=False)
        manager.add_startup_task("success_task", success_task, critical=True)
        
        # Mock configuration and validation
        with patch('app.core.startup.initialize_configuration'), \
             patch('app.core.startup.validate_configuration', return_value=True), \
             patch.object(manager, '_start_model_preloading'):
            
            result = await manager.execute_startup_sequence()
            
            assert result is True
            assert manager.startup_completed is True
            assert success_called is True
    
    @pytest.mark.asyncio
    async def test_task_timeout(self):
        """Test task timeout handling"""
        manager = StartupManager()
        
        async def slow_task():
            await asyncio.sleep(2)  # Longer than timeout
        
        manager.add_startup_task("slow_task", slow_task, critical=True, timeout=1)
        
        # Mock configuration and validation
        with patch('app.core.startup.initialize_configuration'), \
             patch('app.core.startup.validate_configuration', return_value=True):
            
            result = await manager.execute_startup_sequence()
            
            assert result is False
            assert "timed out" in str(manager.startup_errors[0])
    
    @pytest.mark.asyncio
    async def test_shutdown_sequence(self):
        """Test shutdown sequence"""
        manager = StartupManager()
        
        shutdown_called = False
        
        async def shutdown_task():
            nonlocal shutdown_called
            shutdown_called = True
        
        manager.add_shutdown_task("shutdown_task", shutdown_task)
        
        await manager.execute_shutdown_sequence()
        
        assert shutdown_called is True
    
    def test_get_startup_status(self):
        """Test getting startup status"""
        manager = StartupManager()
        
        status = manager.get_startup_status()
        
        assert 'completed' in status
        assert 'errors' in status
        assert 'startup_time' in status
        assert 'model_preload_status' in status


class TestHealthMonitor:
    """Test health monitoring functionality"""
    
    def test_health_monitor_initialization(self):
        """Test health monitor initialization"""
        monitor = HealthMonitor()
        
        assert monitor.component_health == {}
        assert monitor.last_full_check is None
        assert monitor.monitoring_task is None
        assert monitor.metrics_history == []
    
    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring"""
        monitor = HealthMonitor()
        
        # Mock the monitoring loop to avoid infinite loop
        with patch.object(monitor, '_monitoring_loop', new_callable=AsyncMock):
            await monitor.start_monitoring()
            assert monitor.monitoring_task is not None
            
            await monitor.stop_monitoring()
            assert monitor.monitoring_task is None
    
    @pytest.mark.asyncio
    async def test_perform_health_check(self):
        """Test performing health check"""
        monitor = HealthMonitor()
        
        # Mock all check methods
        with patch.object(monitor, '_check_startup_status', new_callable=AsyncMock), \
             patch.object(monitor, '_check_model_status', new_callable=AsyncMock), \
             patch.object(monitor, '_check_gpu_status', new_callable=AsyncMock), \
             patch.object(monitor, '_check_system_resources', new_callable=AsyncMock), \
             patch.object(monitor, '_check_api_status', new_callable=AsyncMock), \
             patch.object(monitor, '_collect_system_metrics', return_value=None):
            
            result = await monitor.perform_health_check()
            
            assert 'overall_status' in result
            assert 'components' in result
            assert 'last_check' in result
            assert 'check_duration_ms' in result
    
    @pytest.mark.asyncio
    async def test_check_startup_status(self):
        """Test startup status checking"""
        monitor = HealthMonitor()
        
        # Mock startup manager
        mock_status = {
            'completed': True,
            'errors': [],
            'startup_time': 5.0
        }
        
        with patch('app.core.health.startup_manager') as mock_startup_manager:
            mock_startup_manager.get_startup_status.return_value = mock_status
            
            await monitor._check_startup_status()
            
            assert 'startup' in monitor.component_health
            component = monitor.component_health['startup']
            assert component.status == HealthStatus.HEALTHY
            assert "completed successfully" in component.message
    
    @pytest.mark.asyncio
    async def test_check_model_status(self):
        """Test model status checking"""
        monitor = HealthMonitor()
        
        # Mock model manager
        mock_model_manager = Mock()
        mock_model_manager.is_ready.return_value = True
        mock_model_manager.get_model_info.return_value = {'is_initialized': True}
        
        with patch('app.core.health.model_manager', mock_model_manager):
            await monitor._check_model_status()
            
            assert 'model' in monitor.component_health
            component = monitor.component_health['model']
            assert component.status == HealthStatus.HEALTHY
    
    @pytest.mark.asyncio
    async def test_check_gpu_status_cpu_mode(self):
        """Test GPU status checking in CPU mode"""
        monitor = HealthMonitor()
        
        with patch('app.core.health.settings') as mock_settings:
            mock_settings.gpu.force_cpu = True
            
            await monitor._check_gpu_status()
            
            assert 'gpu' in monitor.component_health
            component = monitor.component_health['gpu']
            assert component.status == HealthStatus.HEALTHY
            assert "CPU-only mode" in component.message
    
    @pytest.mark.asyncio
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=1)
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.memory_allocated', return_value=1024 * 1024 * 1024)  # 1GB
    async def test_check_gpu_status_gpu_mode(self, mock_memory, mock_props, mock_count, mock_available):
        """Test GPU status checking in GPU mode"""
        monitor = HealthMonitor()
        
        # Mock GPU properties
        mock_props.return_value = Mock(
            name="GTX 1060",
            total_memory=6 * 1024 * 1024 * 1024  # 6GB
        )
        
        with patch('app.core.health.settings') as mock_settings:
            mock_settings.gpu.force_cpu = False
            
            await monitor._check_gpu_status()
            
            assert 'gpu' in monitor.component_health
            component = monitor.component_health['gpu']
            assert component.status == HealthStatus.HEALTHY
    
    @pytest.mark.asyncio
    @patch('psutil.cpu_percent', return_value=50.0)
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    async def test_check_system_resources(self, mock_disk, mock_memory, mock_cpu):
        """Test system resources checking"""
        monitor = HealthMonitor()
        
        # Mock system resources
        mock_memory.return_value = Mock(
            percent=60.0,
            used=4 * 1024**3,  # 4GB
            total=8 * 1024**3  # 8GB
        )
        
        mock_disk.return_value = Mock(
            used=50 * 1024**3,  # 50GB
            total=100 * 1024**3,  # 100GB
            free=50 * 1024**3   # 50GB
        )
        
        await monitor._check_system_resources()
        
        assert 'system' in monitor.component_health
        component = monitor.component_health['system']
        assert component.status == HealthStatus.HEALTHY
    
    @pytest.mark.asyncio
    async def test_check_api_status(self):
        """Test API status checking"""
        monitor = HealthMonitor()
        
        # Mock generation service
        mock_generation_service = Mock()
        mock_generation_service.get_generation_status.return_value = {
            'active_requests': 0
        }
        
        with patch('app.core.health.generation_service', mock_generation_service), \
             patch('app.core.health.settings') as mock_settings:
            mock_settings.api.max_concurrent_requests = 5
            
            await monitor._check_api_status()
            
            assert 'api' in monitor.component_health
            component = monitor.component_health['api']
            assert component.status == HealthStatus.HEALTHY
    
    @pytest.mark.asyncio
    @patch('psutil.cpu_percent', return_value=45.0)
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    async def test_collect_system_metrics(self, mock_disk, mock_memory, mock_cpu):
        """Test system metrics collection"""
        monitor = HealthMonitor()
        
        # Mock system resources
        mock_memory.return_value = Mock(
            percent=60.0,
            used=4 * 1024**3,  # 4GB
            total=8 * 1024**3  # 8GB
        )
        
        mock_disk.return_value = Mock(
            used=50 * 1024**3,  # 50GB
            total=100 * 1024**3,  # 100GB
            free=50 * 1024**3   # 50GB
        )
        
        metrics = await monitor._collect_system_metrics()
        
        assert isinstance(metrics, SystemMetrics)
        assert metrics.cpu_percent == 45.0
        assert metrics.memory_percent == 60.0
        assert metrics.disk_percent == 50.0
    
    def test_determine_overall_health(self):
        """Test overall health determination"""
        monitor = HealthMonitor()
        
        # Test healthy components
        monitor.component_health = {
            'comp1': ComponentHealth('comp1', HealthStatus.HEALTHY, 'OK'),
            'comp2': ComponentHealth('comp2', HealthStatus.HEALTHY, 'OK')
        }
        
        status = monitor._determine_overall_health()
        assert status == HealthStatus.HEALTHY
        
        # Test with one degraded component
        monitor.component_health['comp2'].status = HealthStatus.DEGRADED
        status = monitor._determine_overall_health()
        assert status == HealthStatus.DEGRADED
        
        # Test with one unhealthy component
        monitor.component_health['comp2'].status = HealthStatus.UNHEALTHY
        status = monitor._determine_overall_health()
        assert status == HealthStatus.UNHEALTHY
    
    def test_get_health_summary(self):
        """Test health summary generation"""
        monitor = HealthMonitor()
        
        # Add some test components
        monitor.component_health = {
            'comp1': ComponentHealth('comp1', HealthStatus.HEALTHY, 'OK'),
            'comp2': ComponentHealth('comp2', HealthStatus.DEGRADED, 'Warning'),
            'comp3': ComponentHealth('comp3', HealthStatus.UNHEALTHY, 'Error')
        }
        
        summary = monitor.get_health_summary()
        
        assert summary['overall_status'] == HealthStatus.UNHEALTHY.value
        assert summary['component_count'] == 3
        assert summary['healthy_components'] == 1
        assert summary['degraded_components'] == 1
        assert summary['unhealthy_components'] == 1
    
    def test_metrics_history_management(self):
        """Test metrics history management"""
        monitor = HealthMonitor()
        
        # Create test metrics
        test_metrics = SystemMetrics(
            cpu_percent=50.0,
            memory_percent=60.0,
            memory_used_gb=4.0,
            memory_total_gb=8.0,
            disk_percent=70.0,
            disk_free_gb=30.0
        )
        
        # Update history
        monitor._update_metrics_history(test_metrics)
        
        assert len(monitor.metrics_history) == 1
        assert 'timestamp' in monitor.metrics_history[0]
        assert 'metrics' in monitor.metrics_history[0]
        
        # Test history size limit
        monitor.max_history_size = 2
        for i in range(5):
            monitor._update_metrics_history(test_metrics)
        
        assert len(monitor.metrics_history) <= monitor.max_history_size


class TestComponentHealth:
    """Test ComponentHealth dataclass"""
    
    def test_component_health_creation(self):
        """Test creating ComponentHealth instance"""
        component = ComponentHealth(
            name="test_component",
            status=HealthStatus.HEALTHY,
            message="All good",
            details={'key': 'value'}
        )
        
        assert component.name == "test_component"
        assert component.status == HealthStatus.HEALTHY
        assert component.message == "All good"
        assert component.details == {'key': 'value'}
    
    def test_component_health_to_dict(self):
        """Test converting ComponentHealth to dictionary"""
        component = ComponentHealth(
            name="test_component",
            status=HealthStatus.HEALTHY,
            message="All good",
            last_check=datetime.now()
        )
        
        result = component.to_dict()
        
        assert result['name'] == "test_component"
        assert result['status'] == "healthy"
        assert result['message'] == "All good"
        assert 'last_check' in result


class TestSystemMetrics:
    """Test SystemMetrics dataclass"""
    
    def test_system_metrics_creation(self):
        """Test creating SystemMetrics instance"""
        metrics = SystemMetrics(
            cpu_percent=50.0,
            memory_percent=60.0,
            memory_used_gb=4.0,
            memory_total_gb=8.0,
            disk_percent=70.0,
            disk_free_gb=30.0
        )
        
        assert metrics.cpu_percent == 50.0
        assert metrics.memory_percent == 60.0
        assert metrics.disk_percent == 70.0
    
    def test_system_metrics_to_dict(self):
        """Test converting SystemMetrics to dictionary"""
        metrics = SystemMetrics(
            cpu_percent=50.0,
            memory_percent=60.0,
            memory_used_gb=4.0,
            memory_total_gb=8.0,
            disk_percent=70.0,
            disk_free_gb=30.0,
            gpu_memory_percent=80.0
        )
        
        result = metrics.to_dict()
        
        assert result['cpu_percent'] == 50.0
        assert result['gpu_memory_percent'] == 80.0
        assert len(result) == 10  # All fields


if __name__ == "__main__":
    pytest.main([__file__])
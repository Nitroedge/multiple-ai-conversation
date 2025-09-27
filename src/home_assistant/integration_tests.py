"""
Comprehensive integration tests for the Home Assistant system.
Tests hardware integration, API endpoints, workflows, and security.
"""

import asyncio
import logging
import unittest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json
import tempfile
import os

# Import our modules
from .ha_client import HomeAssistantClient
from .esp32_interface import ESP32Interface, ESP32Device
from .device_manager import DeviceManager
from .automation_engine import AutomationEngine, AutomationRule, AutomationTrigger, AutomationAction
from .environmental_context import EnvironmentalContextManager
from .voice_commands import HomeVoiceCommandProcessor
from .state_monitor import DeviceStateMonitor, MonitoringConfig
from .workflow_engine import WorkflowEngine, WorkflowDefinition, DeviceControlStep, DelayStep
from .security_manager import SecurityManager, SecurityConfig, UserRole, Permission

logger = logging.getLogger(__name__)

class MockHomeAssistantAPI:
    """Mock Home Assistant API for testing"""

    def __init__(self):
        self.states = [
            {
                'entity_id': 'light.living_room',
                'state': 'off',
                'attributes': {'brightness': 0, 'friendly_name': 'Living Room Light'}
            },
            {
                'entity_id': 'sensor.temperature',
                'state': '22.5',
                'attributes': {'unit_of_measurement': '°C', 'friendly_name': 'Temperature Sensor'}
            },
            {
                'entity_id': 'switch.fan',
                'state': 'off',
                'attributes': {'friendly_name': 'Ceiling Fan'}
            }
        ]
        self.service_calls = []

    async def get_states(self):
        return self.states

    async def call_service(self, domain, service, entity_id=None, data=None):
        call = {
            'domain': domain,
            'service': service,
            'entity_id': entity_id,
            'data': data or {},
            'timestamp': datetime.now()
        }
        self.service_calls.append(call)

        # Simulate state changes
        if domain == 'light' and service == 'turn_on' and entity_id:
            for state in self.states:
                if state['entity_id'] == entity_id:
                    state['state'] = 'on'
                    state['attributes']['brightness'] = data.get('brightness', 255)

        return {'success': True}

class MockESP32Device:
    """Mock ESP32 device for testing"""

    def __init__(self, device_id: str):
        self.device_id = device_id
        self.sensors = {
            'temperature': 23.5,
            'humidity': 45.2,
            'motion': False
        }
        self.actuators = {
            'relay1': False,
            'servo1': 0
        }
        self.connected = True

    async def get_status(self):
        return {
            'connected': self.connected,
            'uptime': 12345,
            'memory_free': 32768,
            'wifi_rssi': -45
        }

    async def read_sensor(self, sensor_id):
        if sensor_id in self.sensors:
            return {
                'sensor_id': sensor_id,
                'value': self.sensors[sensor_id],
                'timestamp': datetime.now(),
                'unit': 'various'
            }
        return None

    async def control_actuator(self, actuator_id, action, parameters=None):
        if actuator_id in self.actuators:
            if action == 'set_state':
                self.actuators[actuator_id] = parameters.get('state', False)
            elif action == 'set_position':
                self.actuators[actuator_id] = parameters.get('position', 0)
            return True
        return False

class HomeAssistantIntegrationTests(unittest.IsolatedAsyncioTestCase):
    """Integration tests for the Home Assistant system"""

    async def asyncSetUp(self):
        """Set up test environment"""
        # Create mock APIs
        self.mock_ha_api = MockHomeAssistantAPI()
        self.mock_esp32_device = MockESP32Device("esp32_test_001")

        # Create components
        self.ha_client = HomeAssistantClient("http://localhost:8123", "test_token")
        self.ha_client.session = Mock()
        self.ha_client.session.get = AsyncMock(return_value=Mock(json=AsyncMock(return_value=self.mock_ha_api.states)))
        self.ha_client.session.post = AsyncMock(return_value=Mock(json=AsyncMock(return_value={'success': True})))

        self.esp32_interface = ESP32Interface()
        self.esp32_interface.devices = {"esp32_test_001": self.mock_esp32_device}

        self.device_manager = DeviceManager(self.ha_client, self.esp32_interface)
        self.automation_engine = AutomationEngine(self.device_manager)
        self.environmental_context = EnvironmentalContextManager(self.device_manager)
        self.voice_processor = HomeVoiceCommandProcessor(self.device_manager, self.environmental_context)
        self.state_monitor = DeviceStateMonitor(self.ha_client, self.esp32_interface)
        self.workflow_engine = WorkflowEngine(self.device_manager, self.ha_client, self.esp32_interface)
        self.security_manager = SecurityManager()

    async def asyncTearDown(self):
        """Clean up test environment"""
        if self.state_monitor.monitoring_active:
            await self.state_monitor.stop_monitoring()
        if self.workflow_engine.scheduler_running:
            await self.workflow_engine.stop()

    async def test_ha_client_integration(self):
        """Test Home Assistant client integration"""
        # Test getting states
        states = await self.ha_client.get_states()
        self.assertIsInstance(states, list)
        self.assertTrue(len(states) > 0)

        # Test calling service
        result = await self.ha_client.call_service(
            'light', 'turn_on', 'light.living_room',
            {'brightness': 128}
        )
        self.assertTrue(result.get('success'))

    async def test_esp32_integration(self):
        """Test ESP32 device integration"""
        # Test device discovery
        devices = await self.esp32_interface.discover_devices()
        self.assertTrue(len(devices) > 0)

        # Test sensor reading
        sensor_data = await self.mock_esp32_device.read_sensor('temperature')
        self.assertIsNotNone(sensor_data)
        self.assertEqual(sensor_data['sensor_id'], 'temperature')

        # Test actuator control
        success = await self.mock_esp32_device.control_actuator(
            'relay1', 'set_state', {'state': True}
        )
        self.assertTrue(success)
        self.assertTrue(self.mock_esp32_device.actuators['relay1'])

    async def test_device_manager_integration(self):
        """Test device manager integration"""
        # Test device discovery
        device_count = await self.device_manager.discover_devices()
        self.assertGreater(device_count, 0)

        # Test device control
        success = await self.device_manager.control_device(
            'light.living_room', 'turn_on', {'brightness': 200}
        )
        self.assertTrue(success)

        # Test device status
        devices = await self.device_manager.get_all_devices()
        self.assertIsInstance(devices, list)

    async def test_automation_engine_integration(self):
        """Test automation engine integration"""
        # Create test automation rule
        trigger = AutomationTrigger(
            trigger_id="test_trigger",
            trigger_type="state_change",
            entity_id="sensor.temperature",
            conditions={"above": 25.0}
        )

        action = AutomationAction(
            action_id="test_action",
            action_type="device_control",
            entity_id="switch.fan",
            parameters={"action": "turn_on"}
        )

        rule = AutomationRule(
            rule_id="test_rule",
            name="Temperature Fan Control",
            triggers=[trigger],
            actions=[action]
        )

        # Add rule
        success = await self.automation_engine.add_rule(rule)
        self.assertTrue(success)

        # Test rule execution
        test_trigger = AutomationTrigger(
            trigger_id="manual_test",
            trigger_type="manual",
            entity_id="sensor.temperature"
        )

        await self.automation_engine._execute_rule(rule, test_trigger)

        # Verify rule was added
        rules = await self.automation_engine.get_rules()
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0].rule_id, "test_rule")

    async def test_environmental_context_integration(self):
        """Test environmental context manager integration"""
        await self.environmental_context.start()

        # Simulate environmental data update
        await asyncio.sleep(0.1)  # Allow time for initialization

        # Test comfort calculation
        readings = await self.environmental_context.get_current_readings()
        self.assertIsInstance(readings, dict)

        comfort = await self.environmental_context.get_comfort_level()
        self.assertIsNotNone(comfort)

        await self.environmental_context.stop()

    async def test_voice_command_integration(self):
        """Test voice command processing integration"""
        # Test device control command
        command = await self.voice_processor.process_voice_command(
            "turn on the living room light"
        )
        self.assertIsNotNone(command)
        self.assertEqual(command.entity_id, "light.living_room")
        self.assertEqual(command.action, "turn_on")

        # Execute command
        result = await self.voice_processor.execute_command(command)
        self.assertTrue(result.get('success'))

        # Test environmental query
        query_command = await self.voice_processor.process_voice_command(
            "what's the temperature?"
        )
        self.assertIsNotNone(query_command)

    async def test_state_monitoring_integration(self):
        """Test state monitoring integration"""
        config = MonitoringConfig(health_check_interval=1)
        self.state_monitor.config = config

        await self.state_monitor.start_monitoring()

        # Wait for monitoring to run
        await asyncio.sleep(2)

        # Check device states
        stats = await self.state_monitor.get_system_stats()
        self.assertIsInstance(stats, dict)
        self.assertGreater(stats.get('total_devices', 0), 0)

        # Test manual state update
        await self.state_monitor.update_device_state(
            'test.device', 'on', {'test': True}, 'test'
        )

        device_state = await self.state_monitor.get_device_state('test.device')
        self.assertIsNotNone(device_state)
        self.assertEqual(device_state.state, 'on')

        await self.state_monitor.stop_monitoring()

    async def test_workflow_engine_integration(self):
        """Test workflow engine integration"""
        await self.workflow_engine.start()

        # Create test workflow
        workflow = WorkflowDefinition(
            workflow_id="test_workflow",
            name="Test Workflow"
        )

        # Add steps
        workflow.add_step(DeviceControlStep(
            "step1", "light.living_room", "turn_on", {"brightness": 128}
        ))
        workflow.add_step(DelayStep("step2", 0.1))
        workflow.add_step(DeviceControlStep(
            "step3", "light.living_room", "turn_off"
        ))

        # Register and execute workflow
        self.workflow_engine.register_workflow(workflow)
        execution_id = await self.workflow_engine.execute_workflow("test_workflow")

        # Wait for execution
        await asyncio.sleep(1)

        # Check execution status
        execution = self.workflow_engine.get_execution_status(execution_id)
        self.assertIsNotNone(execution)

        await self.workflow_engine.stop()

    async def test_security_manager_integration(self):
        """Test security manager integration"""
        # Test user creation
        user = await self.security_manager.create_user(
            "testuser", "test@example.com", "TestPass123!",
            UserRole.USER
        )
        self.assertEqual(user.username, "testuser")
        self.assertEqual(user.role, UserRole.USER)

        # Test authentication
        session = await self.security_manager.authenticate_user(
            "testuser", "TestPass123!", "127.0.0.1", "test-agent"
        )
        self.assertIsNotNone(session)

        # Test permission check
        has_permission = await self.security_manager.check_permission(
            session.session_id, Permission.READ_DEVICES, "127.0.0.1"
        )
        self.assertTrue(has_permission)

        # Test invalid permission
        has_admin_permission = await self.security_manager.check_permission(
            session.session_id, Permission.MANAGE_USERS, "127.0.0.1"
        )
        self.assertFalse(has_admin_permission)

        # Test session invalidation
        success = await self.security_manager.invalidate_session(session.session_id)
        self.assertTrue(success)

    async def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        # 1. Start all systems
        await self.state_monitor.start_monitoring()
        await self.workflow_engine.start()
        await self.environmental_context.start()

        # 2. Discover devices
        device_count = await self.device_manager.discover_devices()
        self.assertGreater(device_count, 0)

        # 3. Process voice command
        command = await self.voice_processor.process_voice_command(
            "turn on the living room light to 50% brightness"
        )
        self.assertIsNotNone(command)

        # 4. Execute command through device manager
        result = await self.voice_processor.execute_command(command)
        self.assertTrue(result.get('success'))

        # 5. Create and execute automation workflow
        workflow = WorkflowDefinition(
            workflow_id="end_to_end_test",
            name="End-to-End Test Workflow"
        )

        workflow.add_step(DeviceControlStep(
            "light_on", "light.living_room", "turn_on", {"brightness": 200}
        ))
        workflow.add_step(DelayStep("wait", 0.1))
        workflow.add_step(DeviceControlStep(
            "fan_on", "switch.fan", "turn_on"
        ))

        self.workflow_engine.register_workflow(workflow)
        execution_id = await self.workflow_engine.execute_workflow("end_to_end_test")

        # Wait for workflow completion
        await asyncio.sleep(1)

        # 6. Check results
        execution = self.workflow_engine.get_execution_status(execution_id)
        self.assertIsNotNone(execution)

        # 7. Verify monitoring data
        stats = await self.state_monitor.get_system_stats()
        self.assertGreater(stats.get('total_devices', 0), 0)

        # 8. Cleanup
        await self.environmental_context.stop()
        await self.workflow_engine.stop()
        await self.state_monitor.stop_monitoring()

    async def test_error_handling_and_recovery(self):
        """Test error handling and recovery scenarios"""
        # Test device offline scenario
        self.mock_esp32_device.connected = False

        # Try to control offline device
        success = await self.device_manager.control_device(
            "esp32_test_001", "turn_on"
        )
        self.assertFalse(success)

        # Test invalid voice command
        command = await self.voice_processor.process_voice_command(
            "invalid command that makes no sense"
        )
        self.assertIsNone(command)

        # Test workflow with failing step
        workflow = WorkflowDefinition(
            workflow_id="error_test",
            name="Error Test Workflow"
        )

        # Add step that will fail
        workflow.add_step(DeviceControlStep(
            "invalid_step", "nonexistent.device", "turn_on"
        ))

        self.workflow_engine.register_workflow(workflow)
        execution_id = await self.workflow_engine.execute_workflow("error_test")

        # Wait for execution
        await asyncio.sleep(1)

        execution = self.workflow_engine.get_execution_status(execution_id)
        self.assertIsNotNone(execution)

    async def test_performance_and_scalability(self):
        """Test system performance with multiple operations"""
        # Start monitoring
        await self.state_monitor.start_monitoring()

        # Create multiple concurrent voice commands
        commands = [
            "turn on the living room light",
            "set temperature to 22 degrees",
            "turn off the fan",
            "what's the current temperature?",
            "turn on all lights"
        ]

        # Process commands concurrently
        tasks = [
            self.voice_processor.process_voice_command(cmd)
            for cmd in commands
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        processed_commands = [r for r in results if not isinstance(r, Exception)]

        # Execute valid commands
        for command in processed_commands:
            if command:
                await self.voice_processor.execute_command(command)

        # Create multiple workflows
        workflows = []
        for i in range(5):
            workflow = WorkflowDefinition(
                workflow_id=f"perf_test_{i}",
                name=f"Performance Test {i}"
            )
            workflow.add_step(DelayStep(f"delay_{i}", 0.1))
            workflows.append(workflow)

        # Register and execute workflows concurrently
        for workflow in workflows:
            self.workflow_engine.register_workflow(workflow)

        execution_tasks = [
            self.workflow_engine.execute_workflow(w.workflow_id)
            for w in workflows
        ]

        execution_ids = await asyncio.gather(*execution_tasks)
        self.assertEqual(len(execution_ids), 5)

        # Wait for all executions
        await asyncio.sleep(1)

        # Check all executions completed
        completed_count = 0
        for execution_id in execution_ids:
            execution = self.workflow_engine.get_execution_status(execution_id)
            if execution and execution.status.value in ['completed', 'failed']:
                completed_count += 1

        self.assertEqual(completed_count, 5)

        await self.state_monitor.stop_monitoring()

class TestRunner:
    """Test runner for integration tests"""

    @staticmethod
    async def run_all_tests():
        """Run all integration tests"""
        logger.info("Starting Home Assistant integration tests...")

        # Configure test logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Run tests
        suite = unittest.TestLoader().loadTestsFromTestCase(HomeAssistantIntegrationTests)
        runner = unittest.TextTestRunner(verbosity=2)

        # Run in async context
        loop = asyncio.get_event_loop()
        result = runner.run(suite)

        logger.info(f"Tests completed. Ran {result.testsRun} tests")
        logger.info(f"Failures: {len(result.failures)}")
        logger.info(f"Errors: {len(result.errors)}")

        if result.failures:
            logger.error("Test failures:")
            for test, traceback in result.failures:
                logger.error(f"  {test}: {traceback}")

        if result.errors:
            logger.error("Test errors:")
            for test, traceback in result.errors:
                logger.error(f"  {test}: {traceback}")

        return result.wasSuccessful()

    @staticmethod
    async def run_specific_test(test_name: str):
        """Run a specific test"""
        suite = unittest.TestSuite()
        suite.addTest(HomeAssistantIntegrationTests(test_name))
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return result.wasSuccessful()

# CLI interface for running tests
if __name__ == "__main__":
    import sys

    async def main():
        if len(sys.argv) > 1:
            test_name = sys.argv[1]
            success = await TestRunner.run_specific_test(test_name)
        else:
            success = await TestRunner.run_all_tests()

        sys.exit(0 if success else 1)

    asyncio.run(main())
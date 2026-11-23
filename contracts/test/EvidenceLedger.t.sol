// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Test} from "forge-std/Test.sol";
import {EvidenceLedger} from "../src/EvidenceLedger.sol";

contract EvidenceLedgerTest is Test {
    EvidenceLedger ledger;
    address routerA = makeAddr("routerA");
    address routerB = makeAddr("routerB");
    address stranger = makeAddr("stranger");
    bytes32 opus;
    bytes32 sol;
    bytes32 question = keccak256("What is the capital of France?");

    function setUp() public {
        ledger = new EvidenceLedger(); // this test contract is the owner
        ledger.addRouter(routerA);
        ledger.addRouter(routerB);
        opus = ledger.registerModel("claude-opus-5-5", 4_000_000, 20_000_000);
        sol = ledger.registerModel("gpt-6.1-sol", 2_000_000, 10_000_000);
    }

    // ---- owner-only actions

    function test_OwnerIsDeployer() public view {
        assertEq(ledger.owner(), address(this));
    }

    function test_OnlyOwnerAddsRouters() public {
        vm.prank(stranger);
        vm.expectRevert(EvidenceLedger.NotOwner.selector);
        ledger.addRouter(stranger);
    }

    function test_OnlyOwnerRemovesRouters() public {
        vm.prank(stranger);
        vm.expectRevert(EvidenceLedger.NotOwner.selector);
        ledger.removeRouter(routerA);
    }

    function test_OnlyOwnerRegistersModels() public {
        vm.prank(stranger);
        vm.expectRevert(EvidenceLedger.NotOwner.selector);
        ledger.registerModel("fake-model", 1, 1);
    }

    function test_ModelRegisteredWithPricesAndTime() public view {
        (bool registered, uint64 priceIn, uint64 priceOut, uint64 registeredAt, string memory name) = ledger.models(opus);
        assertTrue(registered);
        assertEq(priceIn, 4_000_000);
        assertEq(priceOut, 20_000_000);
        assertEq(registeredAt, block.timestamp);
        assertEq(name, "claude-opus-5-5");
        assertEq(ledger.modelIdOf("claude-opus-5-5"), opus);
    }

    function test_ModelCannotBeRegisteredTwice() public {
        vm.expectRevert(abi.encodeWithSelector(EvidenceLedger.ModelExists.selector, opus));
        ledger.registerModel("claude-opus-5-5", 1, 1);
    }

    // ---- receipts

    function test_ReceiptEmitsAllFields() public {
        vm.expectEmit(true, true, true, true, address(ledger));
        emit EvidenceLedger.Receipt(question, opus, routerA, 3, 10_000, 11_800_000, 5_990);
        vm.prank(routerA);
        ledger.addReceipt(question, 3, opus, 10_000, 11_800_000, 5_990);
    }

    function test_UnapprovedRouterRefused() public {
        vm.prank(stranger);
        vm.expectRevert(abi.encodeWithSelector(EvidenceLedger.NotRouter.selector, stranger));
        ledger.addReceipt(question, 3, opus, 10_000, 1, 1);
    }

    function test_UnknownModelRefused() public {
        bytes32 fake = keccak256("fake-model");
        vm.prank(routerA);
        vm.expectRevert(abi.encodeWithSelector(EvidenceLedger.UnknownModel.selector, fake));
        ledger.addReceipt(question, 3, fake, 10_000, 1, 1);
    }

    function test_GroupOutOfRangeRefused() public {
        vm.prank(routerA);
        vm.expectRevert(abi.encodeWithSelector(EvidenceLedger.BadGroup.selector, uint8(8)));
        ledger.addReceipt(question, 8, opus, 10_000, 1, 1);
    }

    function test_LastGroupAccepted() public {
        vm.prank(routerA);
        ledger.addReceipt(question, 7, opus, 10_000, 1, 1);
    }

    function test_ScoreAboveMaxRefused() public {
        vm.prank(routerA);
        vm.expectRevert(abi.encodeWithSelector(EvidenceLedger.BadScore.selector, uint16(10_001)));
        ledger.addReceipt(question, 3, opus, 10_001, 1, 1);
    }

    function test_RepeatRefused() public {
        vm.startPrank(routerA);
        ledger.addReceipt(question, 3, opus, 10_000, 1, 1);
        vm.expectRevert(abi.encodeWithSelector(EvidenceLedger.AlreadyReported.selector, question, opus));
        ledger.addReceipt(question, 3, opus, 0, 1, 1);
        vm.stopPrank();
    }

    function test_SameQuestionAllowedForAnotherRouter() public {
        vm.prank(routerA);
        ledger.addReceipt(question, 3, opus, 10_000, 1, 1);
        vm.prank(routerB);
        ledger.addReceipt(question, 3, opus, 0, 1, 1);
    }

    function test_SameQuestionAllowedForAnotherModel() public {
        vm.startPrank(routerA);
        ledger.addReceipt(question, 3, opus, 10_000, 1, 1);
        ledger.addReceipt(question, 3, sol, 10_000, 1, 1);
        vm.stopPrank();
    }

    function test_RemovedRouterRefused() public {
        ledger.removeRouter(routerA);
        vm.prank(routerA);
        vm.expectRevert(abi.encodeWithSelector(EvidenceLedger.NotRouter.selector, routerA));
        ledger.addReceipt(question, 3, opus, 10_000, 1, 1);
    }
}

from typing import Literal

import pytest

from pydentification.training.lightning import callbacks

from .mocks import MockLightningTrainer, StepAheadModule


@pytest.mark.parametrize(
    ["cycle_in_epochs", "max_epochs", "expected"],
    (
        # cycle every epoch
        (1, 10, [True, False, True, False, True, False, True, False, True, False]),
        # cycle every 2 epochs
        (2, 10, [True, True, False, False, True, True, False, False, True, True]),
        # cycle every 3 epochs
        (3, 10, [True, True, True, False, False, False, True, True, True, False]),
        # cycle for entire training
        (5, 5, [True, True, True, True, True]),
    ),
)
def test_cyclic_teacher_forcing(
    cycle_in_epochs: int, max_epochs: int, expected: list[bool], random_prediction_datamodule
):
    trainer = MockLightningTrainer(
        loss_fn=lambda x: float(0),  # mock loss function always returning zero-loss
        max_epochs=max_epochs,
        callbacks=[callbacks.CyclicTeacherForcing(cycle_in_epochs=cycle_in_epochs)],
    )

    random_prediction_datamodule.n_forward_time_steps = 1  # set to 1 before fit, the datamodule is shared between tests
    trainer.fit(StepAheadModule(teacher_forcing=True), datamodule=random_prediction_datamodule)

    assert [epoch["teacher_forcing"] for epoch in trainer.logged_metrics] == expected


@pytest.mark.parametrize(
    ["mock_loss_fn", "max_epochs", "patience", "factor", "max_length", "expected_lengths"],
    (
        # constant loss so trainer sees no improvement
        # 5 epoch patience increases the length by 2 at 6 epoch
        (lambda x: float(0), 10, 5, 2, float("inf"), [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]),
        # linearly decreasing loss, so trainer sees improvement and keeps the length
        (lambda x: float(10 - x), 10, 5, 2, float("inf"), [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        # linearly increasing loss, length is increased every epoch (patience is 1)
        (lambda x: float(x), 10, 1, 2, float("inf"), [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]),
        # loss is dropping every second epoch
        (lambda x: 10 - x - (x % 2), 10, 1, 2, float("inf"), [1, 1, 2, 2, 4, 4, 8, 8, 16, 16]),
        # max length is reached in 6 epoch
        (lambda x: float(x), 10, 1, 2, 32, [1, 2, 4, 8, 16, 32, 32, 32, 32, 32]),
    ),
)
def test_increase_autoregression_length_on_plateau(
    mock_loss_fn,
    max_epochs: int,
    patience: int,
    factor: int,
    max_length: float | int,
    expected_lengths: list[int],
    random_prediction_datamodule,
):
    callback = callbacks.IncreaseAutoRegressionLengthOnPlateau(
        monitor="val_loss",
        patience=patience,
        factor=factor,
        max_length=max_length,
        verbose=True,
    )

    trainer = MockLightningTrainer(
        loss_fn=mock_loss_fn,
        max_epochs=max_epochs,
        callbacks=[callback],
    )

    random_prediction_datamodule.n_forward_time_steps = 1  # set to 1 before fit, the datamodule is shared between tests
    trainer.fit(StepAheadModule(), datamodule=random_prediction_datamodule)

    assert [epoch["n_forward_time_steps"] for epoch in trainer.logged_metrics] == expected_lengths


@pytest.mark.parametrize(
    [
        "cycles",
        "reset_learning_rate",
        "expected_lengths",
        "expected_learning_rates",
        "expected_teacher_forcing",
    ],
    (
        (
            # two cycles in the callback - every second epoch auto-regression length or learning rate is changed
            ["ar_length", "learning_rate"],
            False,  # reset_learning_rate
            [1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 8, 8],
            [1, 1, 1, 1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01],
            [True] * 12,  # teacher forcing is not changed (not given in cycles)
        ),
        (
            # two cycles in the callback - every second epoch auto-regression length or learning rate is changed
            # order is inverted with respect to the previous test, everything else should be the same
            ["learning_rate", "ar_length"],
            True,  # reset_learning_rate
            [1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4],
            [1, 1, 0.1, 0.1, 1, 1, 0.1, 0.1, 1, 1, 0.1, 0.1],
            [True] * 12,  # teacher forcing is not changed (not given in cycles)
        ),
        (
            # callback is called each second epoch, but learning rate is reset at the end of each cycle,
            # so it does not change
            ["ar_length", "learning_rate"],
            True,  # reset_learning_rate
            [1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 8, 8],
            [float(1)] * 12,  # constant learning rate
            [True] * 12,  # teacher forcing is not changed (not given in cycles)
        ),
        (
            # learning rate is changed 3 times, before changing auto-regression length
            ["learning_rate", "learning_rate", "learning_rate", "ar_length"],
            True,  # reset_learning_rate
            [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
            [1, 1, 0.1, 0.1, 0.01, 0.01, 0.001, 0.001, 1, 1, 0.1, 0.1],
            [True] * 12,  # teacher forcing is not changed (not given in cycles)
        ),
        (
            # learning rate is changed 3 times, before changing auto-regression length 2 times
            ["learning_rate", "learning_rate", "learning_rate", "ar_length", "ar_length"],
            True,  # reset_learning_rate
            # auto-regression length is increased at 9-th and 11-th epoch (after learning rate is changed 3 times)
            [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 4, 4],
            # learning rate is decreases until epoch 8 when cycle goes to increasing AR length twice
            # it is reset at epoch 11 - at the end of the cycle
            [1, 1, 0.1, 0.1, 0.01, 0.01, 0.001, 0.001, 0.001, 0.001, 1, 1],
            [True] * 12,  # teacher forcing is not changed (not given in cycles)
        ),
        (
            # teacher forcing toggle is added as the third cycle
            ["learning_rate", "ar_length", "teacher_forcing"],
            False,  # reset_learning_rate
            [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 4, 4],
            [1, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01],
            # teacher forcing is changed at 7-th epoch
            [True, True, True, True, True, True, False, False, False, False, False, False],
        ),
        (
            # learning rate is changed 3 times, before changing auto-regression length and toggling teacher forcing
            ["learning_rate", "learning_rate", "learning_rate", "ar_length", "teacher_forcing"],
            True,  # reset_learning_rate
            [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
            # learning rate is reset to initial value at epoch 11, which is the end of the cycle
            [1, 1, 0.1, 0.1, 0.01, 0.01, 0.001, 0.001, 0.001, 0.001, 1, 1],
            [True, True, True, True, True, True, True, True, True, True, False, False],
        ),
    ),
)
def test_combined_autoregression_callback_cycles(
    cycles: list[Literal["ar_length", "teacher_forcing", "learning_rate"]],
    reset_learning_rate: bool,
    expected_lengths: list[int],
    expected_learning_rates: list[float],
    expected_teacher_forcing: list[bool],
    random_prediction_datamodule,
):
    callback = callbacks.CombinedAutoRegressionCallback(
        cycles=cycles,
        monitor="val_loss",
        patience=2,
        ar_length_factor=2,
        lr_factor=0.1,
        reset_learning_rate=reset_learning_rate,
        max_length=float("inf"),
        verbose=True,
    )

    trainer = MockLightningTrainer(
        loss_fn=lambda x: float(0),
        max_epochs=12,
        callbacks=[callback],
    )

    random_prediction_datamodule.n_forward_time_steps = 1  # set to 1 before fit, the datamodule is shared between tests
    trainer.fit(StepAheadModule(teacher_forcing=True), datamodule=random_prediction_datamodule)

    # only single optimizer with single parameter group is used in the test
    learning_rates = [pytest.approx(epoch["learning_rate_optimizer_0_0"]) for epoch in trainer.logged_metrics]
    assert learning_rates == expected_learning_rates
    assert [epoch["n_forward_time_steps"] for epoch in trainer.logged_metrics] == expected_lengths
    assert [epoch["teacher_forcing"] for epoch in trainer.logged_metrics] == expected_teacher_forcing

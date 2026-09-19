"""Console formatting helpers for training sections and reports."""

from __future__ import annotations

import functools

TOTAL_WIDTH = 50


def section_printer(section_name: str):
    """Decorate a function with a visible section header and footer.

    The decorator prints a centered title, executes the wrapped function, and then
    writes a final separator. This is used for clearer console output during data
    loading and training.

    Args:
        section_name: Title to display in the section header.

    Returns:
        A decorator that wraps the target function and adds the header/footer styling.
    """

    def print_decorator(func):
        """Apply the section header/footer formatting to a callable.

        Args:
            func: Function to wrap.

        Returns:
            The wrapped function with console formatting around execution.
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """Execute the wrapped function while rendering a visible console section.

            Args:
                *args: Positional arguments forwarded to the wrapped function.
                **kwargs: Keyword arguments forwarded to the wrapped function.

            Returns:
                The result returned by the wrapped function.
            """
            section_length = len(section_name)
            remaining_title_width = TOTAL_WIDTH - (section_length + 2)
            if remaining_title_width % 2 == 0:
                left_width = right_width = remaining_title_width // 2
            else:
                left_width = remaining_title_width // 2
                right_width = left_width + 1

            print("=" * left_width + f" {section_name} " + "=" * right_width)
            result = func(*args, **kwargs)
            print("=" * TOTAL_WIDTH)
            return result

        return wrapper

    return print_decorator


if __name__ == "__main__":

    @section_printer("Demo Section")
    def sample_task() -> None:
        """Small example used when running the module directly.

        Returns:
            None. This helper demonstrates the decorator output in a console session.
        """
        print("Executing sample task.")

    sample_task()

/*
Copyright 2023 Daniel Maier

This file is part of ArpackJ (https://github.com/Daniel63656/ArpackJ)

Use of this source code is governed by an MIT-style
license that can be found in the LICENSE file or at
https://opensource.org/licenses/MIT.
*/

package net.scoreworks.arpackj.eig;

public class ArpackException extends RuntimeException {
    private final String msg;

    public ArpackException(String msg) {
        this.msg = msg;
    }

    @Override
    public String getMessage() {
        return msg;
    }
}
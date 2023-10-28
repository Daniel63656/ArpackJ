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